import argparse

import torch
import open_clip
import pandas as pd
from torch.utils.flop_counter import FlopCounterMode
try:
    import fvcore
except:
    fvcore = None

parser = argparse.ArgumentParser(description='OpenCLIP Profiler')

# benchmark specific args
parser.add_argument('--model', metavar='NAME', default='',
                    help='model(s) to profile')
parser.add_argument('--results-file', default='', type=str, metavar='FILENAME',
                    help='Output csv file for results')
parser.add_argument('--profiler', default='torch', type=str, choices=['torch', 'fvcore'])
parser.add_argument('--batch-size', default=1, type=int, help='Batch size for profiling')
parser.add_argument('--image-size', default=224, type=int, help='Batch size for profiling')
parser.add_argument('--text-size', default=77, type=int, help='Batch size for profiling')
parser.add_argument('--no-contrastive', default=False, action="store_true", help='Batch size for profiling')


def profile_fvcore(
        model,
        image_input_size=(3, 224, 224),
        text_input_size=(77,),
        batch_size=1,
        detailed=False,
        force_cpu=False
):
    if force_cpu:
        model = model.to('cpu')
    device, dtype = next(model.parameters()).device, next(model.parameters()).dtype
    example_image_input = torch.ones((batch_size,) + image_input_size, device=device, dtype=dtype)
    example_text_input = torch.ones((batch_size,) + text_input_size, device=device, dtype=torch.int64)
    fca = fvcore.nn.FlopCountAnalysis(model, (example_image_input, example_text_input))
    aca = fvcore.nn.ActivationCountAnalysis(model, (example_image_input, example_text_input))
    if detailed:
        fcs = fvcore.nn.flop_count_str(fca)
        print(fcs)
    return fca.total() / batch_size, aca.total() / batch_size


def profile_torch(model, text_input_size, image_input_size, batch_size=1, force_cpu=False):
    """Profile the full model using torch.utils.flop_counter"""
    if force_cpu:
        model = model.to('cpu')
    device, dtype = next(model.parameters()).device, next(model.parameters()).dtype
    image_input = torch.ones((batch_size,) + image_input_size, device=device, dtype=dtype)
    text_input = torch.ones((batch_size,) + text_input_size, device=device, dtype=torch.int64)

    flop_counter = FlopCounterMode()
    with flop_counter:
        model(image_input, text_input)
    total_flops = sum(flop_counter.get_flop_counts()['Global'].values())
    return total_flops / batch_size


def count_params(model):
    return sum([m.numel() for m in model.parameters()])

def profile_model(model_name, batch_size=1, profiler='torch', image_size=224, text_size=77, no_contrastive=False):
    assert profiler in ['torch', 'fvcore'], 'Only torch and fvcore profilers are supported'
    if profiler == 'fvcore':
        assert fvcore is not None, 'Please install fvcore.'
    model = open_clip.create_model(model_name, force_custom_text=True, pretrained_hf=False)
    model.eval()
    if no_contrastive:
        model.use_contrastive = False 
    if torch.cuda.is_available():
        model = model.cuda()
    
    # if hasattr(model.visual, "image_size"):
        # if isinstance(model.visual.image_size, (tuple, list)):
            # image_input_size = (3,) + tuple(model.visual.image_size[-2:])
        # else:
            # image_input_size = (3, model.visual.image_size, model.visual.image_size)
    # else:
        # image_input_size = (3, 224, 224)
    # text_input_size = (77,)
    # if hasattr(model, 'context_length') and model.context_length:
        # text_input_size = (model.context_length,)
    
    image_input_size = (3, image_size, image_size)
    text_input_size = (text_size,)

    results = {}
    results['model'] = model_name
    results['image_size'] = image_input_size[1]
    model_cfg = open_clip.get_model_config(model_name)
    retries = 2
    while retries:
        retries -= 1
        try:
            results['mparams'] = round(count_params(model) / 1e6, 2)
            results['image_mparams'] = round(count_params(model.visual) / 1e6, 2)
            results['text_mparams'] = round(count_params(model.text) / 1e6, 2)
            if profiler == 'fvcore':
                macs, acts = profile_fvcore(
                    model, image_input_size=image_input_size, text_input_size=text_input_size, force_cpu=not retries, batch_size=batch_size)
                results['gmacs'] = round(macs / 1e9, 2)
                results['macts'] = round(acts / 1e6, 2)
            elif profiler == 'torch':
                total_flops = profile_torch(
                    model, image_input_size=image_input_size, text_input_size=text_input_size, force_cpu=not retries, batch_size=batch_size)
                results['gflops'] = round(total_flops / 1e9, 2)
        except RuntimeError as e:
            print(e)
            raise
    return results


def main():
    args = parser.parse_args()

    # FIXME accept a text file name to allow lists of models in txt/csv
    if args.model == 'all':
        parsed_model = open_clip.list_models()
    else:
        parsed_model = args.model.split(',')

    results = []
    models_with_errors = []
    for m in parsed_model:
        print('='*100)
        print(f'Profiling {m}')
        try:
            row = profile_model(m, text_size=args.text_size, image_size=args.image_size, batch_size=args.batch_size, profiler=args.profiler, no_contrastive=args.no_contrastive)
            results.append(row)
        except Exception as e:
            print(f'Error profiling {m}: {e}')
            import traceback
            traceback.print_exc()
            models_with_errors.append(m)

    df = pd.DataFrame(results, columns=results[0].keys())

    if 'gmacs' in df.columns:
        df = df.sort_values(by=['gmacs', 'mparams', 'model'])
    else:
        df = df.sort_values(by=['gflops', 'mparams', 'model'])

    print('='*100)
    print('Done.')
    print(df)
    if args.results_file:
        df.to_csv(args.results_file, index=False)

    if models_with_errors:
        print('Models with errors:', models_with_errors)


if __name__ == '__main__':
    main()
