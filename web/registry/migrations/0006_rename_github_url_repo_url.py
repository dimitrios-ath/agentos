# Generated by Django 3.2.11 on 2022-02-02 20:32

from django.db import migrations


class Migration(migrations.Migration):

    dependencies = [
        ('registry', '0005_alter_component_repo'),
    ]

    operations = [
        migrations.RenameField(
            model_name='repo',
            old_name='github_url',
            new_name='url',
        ),
    ]
