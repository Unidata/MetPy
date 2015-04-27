c = get_config()

#Export all the notebooks in the current directory to the sphinx_howto format.
c.NbConvertApp.notebooks = ['../notebooks/*.ipynb']
c.NbConvertApp.export_format = 'python'
c.Exporter.template_file = 'python-scripts'
