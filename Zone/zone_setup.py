"""Load an existing Colab/local image or video into the original zone editor."""
from pathlib import Path

MEDIA_SUFFIXES={'.jpg','.jpeg','.png','.bmp','.webp','.mp4','.mov','.avi','.mkv','.m4v','.wmv'}


def load_camera_reference(editor,camera_id,media_path):
    """Create a saved camera or reopen an existing camera without replacing it."""
    if editor.editor_mode!='view':
        raise ValueError('Save or cancel the zone edit first.')
    camera_id=camera_id.strip()
    if not camera_id:
        raise ValueError('Enter a camera name.')
    path=Path(media_path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f'Reference file not found: {path}')
    if camera_id not in editor.config_store.config['cameras']:
        image=editor.read_reference_media(path)
        editor.config_store.add_camera(camera_id,[image.shape[1],image.shape[0]],
                                       editor.encode_reference_png(image),path.name)
    else:
        # Reopening a camera must not silently reinterpret its saved polygons
        # against a different view with the same dimensions.
        camera=editor.config_store.config['cameras'][camera_id]
        if not camera.get('reference_png'):
            raise ValueError('This saved camera lacks an embedded reference. Use the editor to attach its exact original reference.')
    editor.adding_camera=False
    editor.refresh_cameras(camera_id)
    editor.load_camera(camera_id)
    editor.redraw_canvas()
    return camera_id


def display_file_loader(editor,folder='/content'):
    """Pick files already in Colab Files; no laptop upload dialog is opened."""
    import ipywidgets as W
    from IPython.display import display
    directory=W.Text(value=folder,description='Folder:')
    name=W.Text(value='Cam1',description='Camera name:')
    choices=W.Dropdown(description='Reference:',layout=W.Layout(width='650px'))
    refresh=W.Button(description='Refresh files')
    load=W.Button(description='Open camera',button_style='success')
    status=W.Output()
    def refresh_files(_=None):
        with status:
            status.clear_output()
            root=Path(directory.value).expanduser()
            if not root.is_dir():
                choices.options=[]
                print('Folder does not exist:',root)
                return
            choices.options=[(p.name,str(p.resolve())) for p in sorted(root.iterdir())
                             if p.is_file() and p.suffix.lower() in MEDIA_SUFFIXES]
            if not choices.options:
                print('No images/videos found. Upload them through Colab Files, then Refresh files.')
    def open_camera(_):
        with status:
            status.clear_output()
            try:
                if choices.value is None:
                    raise ValueError('Choose a reference file first.')
                camera_id=load_camera_reference(editor,name.value,choices.value)
                print('Opened',camera_id,'- use Add a zone in the editor above.')
            except Exception as error:
                print(type(error).__name__+':',error)
    refresh.on_click(refresh_files);load.on_click(open_camera)
    display(W.VBox([W.HTML('<b>Use a file already in Colab Files</b>'),directory,name,choices,W.HBox([refresh,load]),status]))
    refresh_files()
