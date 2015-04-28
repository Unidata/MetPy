c = get_config()

# Export example notebooks as rst
c.NbConvertApp.notebooks = ['../../../../examples/notebooks/*.ipynb']
c.NbConvertApp.export_format = 'rst'
c.Exporter.template_file = 'examples-rst'
