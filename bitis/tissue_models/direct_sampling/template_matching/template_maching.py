from abc import ABC, abstractmethod


class TemplateMatching(ABC):
    @abstractmethod
    def run(self, template):
        pass
