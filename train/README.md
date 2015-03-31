# Training Programs for Crepe

This directory contains training programs for Crepe. These programs include
* `config.lua`: a unified file for all configurations for dataset, model, trainer, tester and GUI. The current configuration can be used in conjunction with the DBPedia ontology classification dataset distribution. The properties are quite self-exaplanatory, and this should be the starting place if you want to adapt it to different datasets.
* `data.lua`: provide a `Data` class. Both training and validating datasets are instances of this class.
* `main.lua`: the main driver program. This is the only file you should execute.
* `model.lua`: provide a `Model` class. It handles model creation, randomization and transformations during training.
* `mui.lua`: provide a `Mui` class. This class uses `Scroll` class to draw a `nn.Sequential` model in Qt.
* `scroll.lua`: provide a `Scroll` class that starts a scrollable Qt window to drawing text or images. Used only by `Mui`.
* `scroll.ui`: a Qt designer UI file corresponding to the scrollable Qt window.
* `test.lua`: provide a `Test` class. It handles testing, giving you losses, errors and confusion matrices.
* `train.lua`: provide a `Train` class. It handles training with SGD. It supports things like momentum and weight decay.

For more detailed information, please refer to each of the program files.

