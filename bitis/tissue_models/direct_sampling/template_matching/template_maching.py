from abc import ABC, abstractmethod


class TemplateMatching(ABC):
    @abstractmethod
    def run(self, image, template):
        pass
