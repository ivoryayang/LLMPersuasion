import argparse
from tqdm import tqdm
import torch
from torch.nn import DataParallel
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from torch.utils.data import DataLoader
from captum.attr import ShapleyValueSampling, Lime
from captum._utils.models.linear_model import SkLearnLasso

parser = argparse.ArgumentParser(description='Take as input a trained model, a source of input data with raw text and use an attribution method to return a file with two lists: [raw_text, attributions]')
parser.add_argument("-trained_model_path", type=str, help="Path to the trained model")
parser.add_argument("-base_model_name", type=str, default='distilbert-base-uncased', help="name of the base model from hf, needed to load tokenizer")
parser.add_argument("-input_data_path", type=str, help="Input dataset, expected to be a torch dataset")
parser.add_argument("-attribution_method", type=str, default='shap', help="type of attribution method used, currently expecting either shap or lime")
parser.add_argument("-output_file_name", type=str, default='attribution_out.pt', help="output name for the file containing the attributions")

if __name__ == "__main__":
    args = parser.parse_args()

    #load model and distribute across gpus via data parallel
    model = AutoModelForSequenceClassification.from_pretrained(args.trained_model_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_gpus = torch.cuda.device_count()
    device_ids = [i for i in range(num_gpus)]
    model = DataParallel(model, device_ids=device_ids)
    model = model.to(device)

    # local wrapper func to coerce output of model into right "format" for captum
    # i.e. a batch x num_class tensor
    def forward_func_wrapper_for_sequence_classification(input_ids, attention_mask):
        logits = model(input_ids=input_ids, attention_mask=attention_mask)['logits']
        probs = torch.nn.functional.softmax(logits, dim=-1)
        return probs

    # tokenizer = AutoTokenizer.from_pretrained(args.base_model_name)
    input_data = torch.load(args.input_data_path)
    input_dataloader = DataLoader(input_data, batch_size=1, shuffle=False) #batchsize = 1 because shap

    raw_texts, all_input_ids, attributions = [], [], []
    print(f'Attributing feature importance with {args.attribution_method}...')
    for i, batch in enumerate(tqdm(input_dataloader)):
        batch_labels = batch['labels']
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        raw_text_i = batch['raw_text']
        feature_masks = batch['feature_masks'].to(device)
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)


        #different parameters for different attributions methods
        if args.attribution_method == 'shap':
            method = ShapleyValueSampling(forward_func_wrapper_for_sequence_classification)
            #change the parameters of attribute if you want better results at the cost of time
            #see https://captum.ai/api/shapley_value_sampling.html
            attr_i = method.attribute(input_ids, additional_forward_args=(attention_mask,),
                                      target=1, n_samples=2, perturbations_per_eval=10,
                                      feature_mask=feature_masks)
        elif args.attribution_method == 'lime':
            method = Lime(forward_func_wrapper_for_sequence_classification,
                          interpretable_model=SkLearnLasso(alpha=3e-5))
            #change the parameters of attribute if you want better results at the cost of time
            #see https://captum.ai/api/lime.html
            attr_i = method.attribute(input_ids, additional_forward_args=(attention_mask,),
                                      target=0, n_samples=50, perturbations_per_eval=10,
                                      feature_mask=feature_masks)
        else:
            print(f'attribution method ({args.attribution_method}) not defined!')
            pass

        raw_texts.extend(raw_text_i)
        all_input_ids.append(input_ids)
        attributions.append(attr_i)

    attributions, all_input_ids = torch.cat(attributions), torch.cat(all_input_ids)

    output_file = {"raw_text":raw_texts,
                   "all_input_ids":all_input_ids,
                   "attributions": attributions,
                   'attribution_method': args.attribution_method,
                   "base_model_name":args.base_model_name}

    torch.save(output_file, args.output_file_name)

    print(f'finished!')