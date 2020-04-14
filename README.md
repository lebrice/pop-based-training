# (WIP) Simple Implementation of [Population-Based Training for Loss Function Optimization](https://arxiv.org/abs/2002.04225)

This paper describes an approach to train the model and optimize the hyperparameters at the same time using an evolutionary algorithm. It is heavily based on the [Population Based Training paper](https://arxiv.org/abs/1711.09846), with an added parameterization of the loss function (which I have not yet implemented).

## Usage
Check out the [dummy example file](dummy_example.py) file for an explanation. Basically, you should implement your own hyperparameter class, and have it subclass the `HyperParameters`. This way, the attributes of that class will become dynamic and have their values changed during training.

Then, create an evaluation function, which should take in an optional `Candidate` instance as an argument, which contains a previously-trained model as well as new hyperparameters, and constructs a new model, loads as much of the state from the old model (if provided) as possible, and evaluates this new model. The result of this evaluation is the "fitness" value (increasing). The evaluation function should then return a new `Candidate` instance.

That's basically it! Now just call the `epbt()` function, passing in the right arguments, and you're good to go.


## Example
For a concrete example, see [the `mnist_example.py` file](mnist_example.py).


## Notes
- This is implemented without any requirements for

## Requirements:
I was having fun playing around with Python3.8 when writing this, so it is a requirement for now.
It would be pretty easy to change a few things and bring it down to 3.7. If anyone would like that, let me know.
In the mnist example, I also make use my own [`simple-parsing`](https://github.com/lebrice/SimpleParsing)
package to generate the command-line arguments, but it isn't required by any means.

```console
conda env create --file environment.yml
```

