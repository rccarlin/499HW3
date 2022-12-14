# Implementation Choices
## (or: why my code looks this funky)

### Encoding
I didn't want to take too many unnecessary loops through the dataset, so I decided to append each episode's commands, 
actions, and targets together as I encoded them. At the time of writing this function, my brain thought it was
more intuitive to have the output be [[episode_actions], [episode_targets]] instead of a list of tuples. 
I may have thought this would be easier to index through for calculating metrics and such (but honestly I forgot why I 
did this). I don't see any reason why this method can't work, but it did cause some issues down the line. I  had to
tweak utils.prefix_match() to adjust for the fact that actions and targets were separate, because I knew changing that
function would be easier than me restarting this entire program.

I also did the padding by hand because I didn't immediately understand how packing worked. I just kept track of the
longest episode and the most tasks per episode and then added padding after everything had been encoded. I believe this
worked because both of these maxes (sp?) occur in the training set.

Also I added padding/ start/ end tokens for actions and targets in the utils file because they made the most sense to me
there, I hope that's okay.

I remembered to make batches this time!! It's the little things.


### Setup model
Hidden dimension 2 and embedding dimension 128 because that's what I've used in the past and it was fine.

### Setup optimizer
I used cross entropy because that's what we've been using and tutorials online also used it so I thought it would be 
okay to sick with what I know. Plus I knew cross entropy had a way to gracefully handle padding (which there is a lot 
of), and I wanted to make use of that. Whether or not I did everything I needed to have it stop at/ not care about padding
is a different question I don't have the answer to.

I used an Adam optimizer because that's what 9/10 tutorials used, and I don't know enough about optimizers to know if that
was a good choice or not. It didn't seem to crash the program, so I kept it.

lr = .05 because it's my emotional support learning rate


### The Model
A formal ~~apology~~ rant about the model: I spent over a week on it before realizing (today, 11/16) that I was 
using the lstm's hidden state to calculate loss and make predictions instead of the lstm's output (confusingly
also a type of hidden state). I was changing things from hidden to out when I realized I didn't have the correct initial
starting point. Okay, so I went to fix that but then the shapes of things were all wonky because the LSTM was spitting out 
a 3D shape when I needed a 2D. With the deadline fast approaching, I decided code that compiles is better than code I 
know is wrong. So I made a new fully connected layer to make the LSTM output 2D, and then changed the first dim of 
the action and target fully connected layers from the hidden dimension (which by now has been squashed out) to the length
of the onehot vector. Why did I have to do this? I assume something is wrong because this feels like too many layers...
In the end, this is what it took to get the LSTM output into the action and target outputs.

Choices made:
- LSTM because I thought it would be easier to use than the other options since I had used one for HW1... it could have
  been worse but I did have to relearn everything about LSTMs so it's not like using one gave me a head start on this 
- I made a onehot vector of the decoder input because it was recommended and all of my previous attempts crashed because
  the shape was wrong. It took me a little bit to figure out how to pull the necessary outputs for teacher forcing, but
  it was fun to experiment with indexing lol
- I used a simple dot-product flat attention because I only had a day to work on it, given how long the base model took
  me. I couldn't find many examples of how to do attention with torch and LSTMs, and many of the ones I did find used 
  layers I didn't have time to figure out how to use (dimension sizes will be the death of me). I'm 80% sure this attention
  isn't good because there's no weights or anything, but I tried.
- No transformers because I ran out of time. I did look at tutorials on how to do it but none of them integrated well 
  with what I currently had. Plus, I thought adding the transformer was as simple as swapping it out for the encoder,
  all the implementations I saw were confusing and not simple. tbh, It's probably not that hard but the lack of sleep
  and the 2 week long stress headache this hw has given me has turned my brain to mush. And like, this is definitely 
  a me problem, I'm sure this hw would go a lot faster if I had more ML experience
- Added student forcing (was I supposed to?), if I had more time I would have been interested in seeing the
  difference in performance


### Training loops and such
I put the validation calculations (which we were told to only do on validation) in an if statement in train_epoch() as 
opposed to in validate() because otherwise I would need to have train_epoch() return the model's output unnecessarily.

I used stack() when turning the model's outputs into predictions because I didn't put action and targets together. I 
had fun trying to figure out argmax and the other torch functions for this, I didn't feel totally out of my element lol.

I used the given prefix_match (hopefully correctly, I kinda wasn't sure what we were supposed to do here). If I had 
had more time or thought it would be extra credit, I would have done longest common subsequence because I think that's 
a more fair measure of accuracy. I may still do that because I feel bad about not finishing most of this assignment.

Edit: in the hour before I need to turn it in, I have added various accuracy functions of varying helpfulness. I ended
up not doing the normal LCS because the sequences don't have to start at the same time step (which could reward 
predictions that are way too late/ accidentally the same action/target but for a different episode).
So I made mod[ified]LCS that tracks the longest common subsequence that happen at the same time. This redeems
predictions that start off bad but get back on track.

I thought it would be fun to come up with other easy calculations that could still kind of work, so I did.
samePlace() tracks how many correct predictions happen, regardless of the accuracy of predictions made before or after.
youTried() tracks how many partially correct predictions happen, regardless of ...
Although using this function to measure accuracy *seems* like a cop-out, it actually makes sense when you consider the
fact that some of these "correct" actions/ targets are not always intuitive to humans (so imagine what the computer
must be thinking). <-- example: put knife in fridge is supposed to result in putObject, fridge, but aren't we putting
the knife?

For all three of these metrics, I had to come up with a way that didn't consider padding (since there's so much padding,
I didn't want relevant matches/ disagreements to get dominated by matching padding). At first I had the code break when
the actual action began to have padding, but then I though that didn't penalize predicting too many actions. 



Sorry no outputs to analyze. I just finished fixing the base model at 4:00 today and my code is too slow to get results
by midnight. 

I know before I fixed the base, the loss was garbage because my model kept outputting the same thing over and over.

Again, very sorry.