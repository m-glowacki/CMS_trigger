from tensorflow import keras
import hls4ml
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
import os

def hls4ml_converter(model, directory):
        
    config = hls4ml.utils.config_from_keras_model(model, granularity='model')

    print("-----------------------------------")
    print("Configuration")
    print(config)
    print("-----------------------------------")

    hls4ml.model.optimizer.OutputRoundingSaturationMode.layers = ['Activation']
    hls4ml.model.optimizer.OutputRoundingSaturationMode.rounding_mode = 'AP_RND'
    hls4ml.model.optimizer.OutputRoundingSaturationMode.saturation_mode = 'AP_SAT'
   
    hls_model = hls4ml.converters.convert_from_keras_model(model,
                                                           hls_config=config,   
                                                           output_dir=directory,       
                                                           clock_period=(1/.24), 
                                                           io_type='io_stream')     
    
    hls_model.compile()
    return hls_model


def main(model_path, outdir, read_vivado):
    os.makedirs(outdir, exist_ok=True)
    model = keras.models.load_model(model_path)
    hls_model = hls4ml_converter(model, outdir)
    hls_model.build(csim=False)

    if read_vivado:
        hls4ml.report.read_vivado_report(outdir)

if __name__ == "__main__":
    parser = ArgumentParser(description=__doc__, formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument("-m","--model_path" , nargs="+", help="Path to pickled training data")
    parser.add_argument("-o","--outdir" , nargs="+", help="path to output dir of model and graph")
    parser.add_argument("-v", "--vivado", action="store_true", help = "print report?")
    args = parser.parse_args()
    main(args.model_path[0], args.outdir[0], args.vivado)