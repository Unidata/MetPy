{% if fullname in ['metpy.calc', 'metpy.plots.ctables', 'metpy.xarray'] %}

{% include 'overrides/' ~ fullname ~ '.rst' with context %}

{% else %}
   {% if name == 'io' %}
      {% set nice_name = 'Reading Data' %}
   {% else %}
      {% set nice_name = name | title | escape %}
   {% endif %}

{{ (nice_name ~ ' ``(' ~ fullname ~ ')``')|underline }}

.. automodule:: {{ fullname }}

   {% block attributes %}
   {% if attributes %}
   .. rubric:: {{ _('Module Attributes') }}

   .. autosummary::
   {% for item in attributes %}
      {{ item }}
   {%- endfor %}
   {% endif %}
   {% endblock %}

   {% block functions %}
   {% if functions %}
   .. rubric:: {{ _('Functions') }}

   .. autosummary::
      :toctree: ./
   {% for item in functions %}
      {{ item }}
   {%- endfor %}
   {% endif %}
   {% endblock %}

   {% block classes %}
   {% if classes %}
   .. rubric:: {{ _('Classes') }}

   .. autosummary::
      :toctree: ./
   {% for item in classes %}
      {{ item }}
   {%- endfor %}
   {% endif %}
   {% endblock %}

   {% block exceptions %}
   {% if exceptions %}
   .. rubric:: {{ _('Exceptions') }}

   .. autosummary::
      :toctree: ./
   {% for item in exceptions %}
      {{ item }}
   {%- endfor %}
   {% endif %}
   {% endblock %}

{% endif %}
