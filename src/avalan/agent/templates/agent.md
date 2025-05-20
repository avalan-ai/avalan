{% block role %}
    {% if roles %}
        {% for role in roles %}
            {{ role }}
        {% endfor %}
    {% elif role %}
        {{ role }}
    {% else %}
        You are a helpful assistant.
    {% endif %}

    {% if name %}
        Your name is {{name}}.
    {% endif %}
{% endblock %}

{% block task %}
    {{ task }}
{% endblock %}

{% block instructions %}
    {% if instructions %}
        {% for instruction in instructions %}
            {{ instruction }}
        {% endfor %}
    {% endif %}
{% endblock %}

{% block rules %}
    {% if rules %}
        {% for rule in rules %}
            {{ rule }}
        {% endfor %}
    {% endif %}
{% endblock %}
