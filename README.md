## Task description
The task is to maximize the win rate playing the game of [Hangman](https://poki.com/en/g/hangman) with 6 lives, i.e., if you make 6 incorrect letter guesses, you lose. \
The provided file `words_250000_train.txt` contains a total of $227,300$ words made of small alphabet
letters, used for training the model. A completely disjoint set of words is used for testing.

## Strategy used
I trained a bidirectional LSTM neural network created using PyTorch locally. Training was performed on a custom dataset created from the given word dictionary. Given a state of the game (i.e. all letters guessed and revealed yet), the model tries to predict the best letter to make the next guess. The best model from all iterations is used for predictions.

### Dataset creation -
- For every word in the given dictionary, I created $15$ random data points of ‘guesses made in the past’.
- Due to creating 15 different guesses per word, we have over **3.4 million data points** to train on.
- **Scalability**: This idea can be scaled for creating upto $\mathcal{O}(2^n)$ data points for a $n$-length word.
- Every generated set of guesses has the following realistic game conditions
    - at least 1 letter remaining to guess correctly in the word
    - at most 5 incorrectly guessed letters (at most 5 lives lost)
- One-hot encode the ‘masked word’ of length $n$ containing `__` (empty) symbols for un-guessed letters.
    - as a PyTorch tensor of size $n \times 27$,
    - where each of the $n$ rows has $1$ at `ord(letter) - 97` or at index `26` for `__`
    - `0` everywhere else.
- The ideally expected output of the model is an equal probability vector for all remaining letters in the word.


### Model architecture -
- A bidirectional LSTM that first takes as input a ‘masked word’ encoded as above.
    - LSTM with parameters `num_layers = 3, hidden_size = 256, dropout = 0.2`.
    - This LSTM layer gives output of size $512$, which is `2 * hidden_size` due to bidirectional model.
- Add a dropout layer of $0.4$ before connecting this to a linear layer.
- Concatenate a 1-0 indicator tensor of size $26$ indicating all guesses made yet.
- This tensor of size $512+26$ is passed to a linear layer of dimensions $(538,128)$.
- Another dropout layer of $0.4$ between fully connected linear layers.
- Finally a linear layer of size $(128,26)$, which gives 'next guess' probabilities for all the letters.

### Loss criterion - 
We use `nn.CrossEntropyLoss`, since this uses a softmax function to compare with labels.

### Training parameters -
- Random $9:1$ ratio data split for training and validation, `batch_size = 500`.
- ADAM Optimizer with initial `learning_rate = 1e-3, weight decay = 1e-5`
- StepLR scheduler that cuts learning rate by factor of $\frac{1}{2}$ every $10$ epochs.
- Trained for a total of 250 epochs.

### Final model and guess selection -
- Model at epoch $241$ had the least validation loss, and was selected for final predictions.
- The highest probability un-guessed letter from the model is used as the next guess.

## Important files
1. `Hangman_models_training.ipynb` - Dataset creation and model training locally.
2. `Hangman_run_experiments.ipynb` - All the necessary functions to play Hangman games after loading a pre-trained model on random or chosen words. 
3. `Hangman_Strategy_Aditya_Raut.pdf` - A PDF file with description of strategy

Very large sized outputs in the training Jupyter notebook are cleared for better readability.