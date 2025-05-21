from prefect import task

@task(name='export-model', description='Exporting model into SavedModel Format')
def export_model(MODELS, dir_path) -> None:
    MODELS.export(dir_path)