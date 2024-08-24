from abc import ABC, abstractmethod

# Define an abstract base class for plugins
class Plugin(ABC):
    @abstractmethod
    def run(self, frame):
        pass  # To be implemented by child classes