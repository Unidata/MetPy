c = get_config()

# Output all example notebooks as Python scripts
c.NbConvertApp.notebooks = ['../notebooks/*.ipynb']
c.NbConvertApp.export_format = 'python'
c.Exporter.template_file = 'python-scripts'
