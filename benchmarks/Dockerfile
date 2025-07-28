FROM python:3.12

RUN pip install --no-cache-dir netcdf4 asv pysu metpy

COPY --chmod=700 entrypoint.sh /

ENTRYPOINT ["/entrypoint.sh"]
