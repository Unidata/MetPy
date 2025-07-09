FROM python:3.12

RUN pip install --no-cache-dir matplotlib netcdf4 numpy pandas pint pooch pyproj scipy traitlets xarray asv pysu metpy

COPY --chmod=700 entrypoint.sh /

ENTRYPOINT ["/entrypoint.sh"]
