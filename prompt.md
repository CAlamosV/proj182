I have a dataset of N sequences of lenth L of HxW black and white images. 
Each image has a single black pixel.
Each sequence represent the path of a black pixel bouncing on the walls of a HxW box.

I want to train a transformer to predict the next position of the black pixel given the previous positions.
That is, I want it to predict a tuple (x,y) given a sequence of images.
I want the training to be done as in next-token prediction, that is, the model should predict the next position given the previous images.
In particular, a sequence of L images should contain L-1 training examples.

I want the loss to be the euclidean distance between the predicted position and the true position.

The transformer architecture used should be simple and relatively easy to train from scratch.

Please write code to train the model. Use whichever framework is simplest to use.

I have a function generate_random_sequence() that returns a tuple of  (a list of L images of shape HxW, a list of L positions tuples (x,y)).
Use it to generate the training data.

TODO
- First target something like 4th image?
- sequences should have consecutive pixels more spread out
- new loss to penalize predicing previous frame?
- curriculum leaning?
- Tune architecture?