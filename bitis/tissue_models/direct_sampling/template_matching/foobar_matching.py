from .template_maching import TemplateMatching


class FooBarMatching(TemplateMatching):
    def __init__(self):
        super().__init__()
        self.template_matchers = {}

    def run(self, template, selector, **kwargs):
        return self.template_matchers[selector].run(template, **kwargs)
