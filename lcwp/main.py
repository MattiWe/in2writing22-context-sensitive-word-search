import click
from lcwp.data.cloth import compile_cloth
from lcwp.data.singlechoice import compile_singlechoice
from lcwp.data.netspeak import compile_netspeak
from lcwp.evaluation.datasets import dataset_statistics
from lcwp.models.pretrained import predict_from_pretrained
from lcwp.models.finetune import finetune
from pathlib import Path


def elsenone(var):
    return None if var is None else Path(var)

@click.group()
def cli():
    pass


@cli.group()
# @click.option('--epochs', type=click.INT, help='Number of epochs the models should be trained for', default=30,
#               show_default=True)
# @click.option('--max_seq_length', type=click.INT, help='Length of the input sequences (in tokens)', default=50,
#               show_default=True)
# @click.option('--cv', type=click.Path(), help='Number of cross-validation iterations', default=5, show_default=True)
# @click.option('--cache_dir', type=click.Path(), help='Path where to create the cache (i.e. for tensorflow hub models)',
#               default='./cache', show_default=True)
# @click.option('-o', '--output_path', type=click.Path(), help='Path where the results will be written to',
#               default='./output', show_default=True)
# @click.option('-e', '--export_model', type=click.BOOL, help='If the trained models should be exported', is_flag=True,
#               default=False, show_default=True)
def data():
    """ Functions to generate low-context queries from source text. """
    pass


@data.command()
@click.option('-o', '--output_path', type=click.Path(), help='Path where the output will be written to.',
              default='./output/cloth-queries', show_default=True)
@click.option('--cloth_path', type=click.Path(), help='Path to the CLOTH corpus.')
def cloth(output_path, cloth_path):
    """ Compile the teachers queries form the cloze test dataset."""
    op = Path(output_path)
    op.mkdir(parents=True, exist_ok=True)
    compile_cloth(op, elsenone(cloth_path))


@data.command()
@click.option('-o', '--output_path', type=click.Path(), help='Path where the output will be written to.',
              default='./output/cloth-queries', show_default=True)
@click.option('-i', '--input_file', type=click.Path(), help='Input file in the .ndjson format we use for LCWP')
@click.option('-n', '--output_name', type=str, help='The name of the output .ndjson with the results')
@click.option('-l', '--query_length', type=click.INT, help='Length of the queries to be considered', default=0)
def netspeak(output_path, output_name, input_file, query_length):
    """ Get the answers to the given queries from Netspeak and save them as a test dataset. """
    op = Path(output_path)
    op.mkdir(parents=True, exist_ok=True)
    compile_netspeak(op, output_name, Path(input_file), query_length)


@data.command()
@click.option('-o', '--output_path', type=click.Path(), help='Path where the output will be written to.',
              default='./output/singlechoice-queries', show_default=True)
@click.option('--nyt_path', type=click.Path(), help='Path to the New York Times corpus. Will be skipped if empty.')
@click.option('--wikitext_path', type=click.Path(), help='Path to the Wikitext dataset. Will be skipped if empty.')
@click.option('--europarl_path', type=click.Path(), help='Path to the Europarl corpus. Will be skipped if empty.')
@click.option('-t', '--training', type=click.BOOL,help='Generate the training data', is_flag=True, default=False,
              show_default=True)
def singlechoice(output_path, nyt_path, wikitext_path, europarl_path, training):
    """ Compile the low-context MLM inputs to fine-tune language models """
    op = Path(output_path)
    op.mkdir(parents=True, exist_ok=True)

    compile_singlechoice(op, elsenone(nyt_path), elsenone(wikitext_path), elsenone(europarl_path), training)

@data.command()
@click.option('-o', '--output_path', type=click.Path(), help='Path where the output will be written to.',
              default='./output/data-evaluation', show_default=True)
@click.option('-i', '--input_file', type=click.Path(), help='Input file in the .ndjson format we use for LCWP')
def evaluate(output_path, input_file):
    """ Generate statistics of the datasets. """
    op = Path(output_path)
    op.mkdir(parents=True, exist_ok=True)

    dataset_statistics(op, input_file)


@cli.command()
@click.option('-o', '--output_path', type=click.Path(), help='Path where the output will be written to.',
              default='./tuned-models', show_default=True)
@click.option('--training_file', type=click.Path(), help='File with the training examples in the .ndjson format we use for LCWP')
@click.option('--validation_file', type=click.Path(), help='File with the validation examples in the .ndjson format we use for LCWP')
@click.option('--model_name', help='The name of the model from huggingface or the directory of the saved checkpoints.')
@click.option('--strategy', help='Training strategy (mlm, clminfill, s2sranked)', default='mlm')
def train(output_path, training_file, validation_file, model_name, strategy):
    """ Functions to train (fine-tune) models on low-context queries  """
    op = Path(output_path)
    op.mkdir(parents=True, exist_ok=True)
    finetune(op, model_name, strategy, training_file, validation_file)


@cli.command()
@click.option('-o', '--output_path', type=click.Path(), help='Path where the output will be written to.',
              default='./output/cloth-queries', show_default=True)
@click.option('-i', '--input_file', type=click.Path(), help='Input file in the .ndjson format we use for LCWP')
@click.option('--batch_size', type=click.INT, help='How many queries to test in one pass to the model', default=30)
@click.option('--model_name', help='The name of the model from huggingface or the directory of the saved checkpoints.')
@click.option('--task', help='Which task to solve: mlm, clm, bart, infill, bart-infill.', default='mlm')
@click.option('--min_query_length', type=click.INT, help='Minimum length of queries to be considered', default=3)
@click.option('-l', '--max_query_length', type=click.INT, help='Maximum length of queries to be considered. Default 0 means all of them', default=9)
def test(output_path, input_file, batch_size, model_name, task, min_query_length, max_query_length):
    """ Functions to run models on low-context queries and save the results """
    op = Path(output_path)
    op.mkdir(parents=True, exist_ok=True)
    predict_from_pretrained(op, model_name=model_name, model_type=task, input_dataset=Path(input_file), device_id=0,
                            batch_size=batch_size, min_query_length=min_query_length, max_query_length=max_query_length)


@cli.command()
def evaluate():
    """ Functions to compute the evaluation metrics based on the input files """
    click.echo('Not implemented yet.')


if __name__ == "__main__":
    cli()
