import click


@click.group()
def cli():
    pass


@cli.command()
@click.argument('directory_path', type=click.Path(exists=True, file_okay=False))
def train_emg(directory_path: str):
    from decision_trees.datasets.emg_raw import EMGRaw

    emg_loader = EMGRaw(directory_path)


if __name__ == '__main__':
    cli()
