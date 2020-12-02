import os
import click
from experimentset import ExperimentSet


@click.group()
def cli():
	pass


@cli.command('experiment', help='Experiment mode')
@click.option('--file', '-f', required=True, help=u'File that contains the experiments that will be executed.')
@click.option('--gpu', '-g', required=False, default="0", help=u'GPU index')
def experiment(file, gpu):
	# Set visible GPU
	os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
	os.environ["CUDA_VISIBLE_DEVICES"] = gpu

	experimentSet = ExperimentSet(file)
	experimentSet.run_all()


if __name__ == '__main__':
	cli()