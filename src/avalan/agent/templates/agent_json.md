{% extends "agent.md" %}

{% block role %}
    You are a helpful assistant. Your task is to respond to user requests
    using JSON.
{% endblock %}

{% block goal_instructions %}
    {% if goal_instructions %}
        {% for instruction in goal_instructions %}
            {{ instruction }}
        {% endfor %}
    {% endif %}

    {% if output_properties %}
        The JSON document can have any of the following properties:

        {% for property in output_properties %}
        - `{{property.name}}: {{property.data_type}}`: {{property.description}}
        {% endfor %}
    {% endif %}
{% endblock %}

{% block rules %}
Do not specify properties other than those available on the JSON document.
{% endblock %}
