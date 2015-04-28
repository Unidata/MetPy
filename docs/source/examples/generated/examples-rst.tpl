{%- extends 'rst.tpl' -%}

{%- block header -%}
{{ ("=" * (resources['metadata']['name'] | length)) }}
{{resources['metadata']['name'].replace('_', ' ')}}
{{ ("=" * (resources['metadata']['name'] | length)) }}
{%- endblock header -%}
