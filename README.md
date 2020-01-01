#### Simulating invoice generation process  

This repository displays a process we went through to create a simulated business documents for fast experimentation.
All experiments could be run as `python run_experiments.py command_name`, where command name is the name of the experiment.
Simplest experiments are provided as tests in `concepts_test.py`.  
The code contains codes for model from our article https://arxiv.org/abs/1904.12577 together with generators code and some easier models
(using tf.data.dataset and tensorpack dataflows) possibly enabling fast experimentation. 

##### Introduction

Not only our artificial networks and models, even we have observed a lot of business documents and invoices so far.
Thus we came up with an idea that could potentially save a lot of time and allow us to understand the problem of data extraction more.  
To try to simulate the generative process of business documents.  

##### The base assumptions

Gathering all the layouts and business document generators is out of anybody's possibility, our idea was much simpler.  
First we usually do not want to extract evary single detail, but only the data important to the user/customer.  
And second, all the text data to be extracted are usually located in a proximity or some understandable relation with other texts, that directly or indirectly explain
the class and meaning. Example being a header that tells us that all amounts in a column will be the amounts to be paid.  
  
In other words - each target textbox, comes with some number of textboxes that together state the meaning (class)
 of the information to be extracted. So focusing only on the important data to extract, we can ommit most of text we usually see in the document.
  
In our simulated abstraction each box has some class, position and size.
A target 'concept' is a combination of textboxes, where exactly one textbox has known target class.
Each of those properties is additionally a random variable and can come from any distribution, as we please.
A page of our artifical document is then constructed as a combination of more concepts,
 that can appear with some probability in a random number of cases.

##### The journey of experiments

The path towards the full simulated data extraction has more steps, that would validate our approach.
First we have verified, that a simple network can predict simple puzzles, like deciding a class based on a distance or position.
(See `concepts_test.py`)  
Then we moved to a more realistic generator based on our experience and visual inspection and started easily.
We let the network know which boxes should be groupped together and gained insight on the network's size to predict the target values with 
more than 0.9 f1 score (`run_experiments.py: fixed_known_borders, fixed_known_borders_bigger`).  
When we added some random shuffling and predicting also non-interesting boxes (`fixed_known_borders_all_boxes_noshuffle,
fixed_known_borders_bigger_all_boxes_noshuffle, fixed_known_borders_all_boxes_shuffle`), the score stayed near at 0.88 f1.  

Then came an important step - we hid the group information from the network and pass the information only in a text 
reading order in a page exactly as we did in our first article. Including neighbours for each box.
Here we let compete two models - the model from article and a simple baseline (`articlemodel, baseline_rendered`).
To our surprise, even after simplifying the generated problems and tuning the class weights, the simple baseline performed better
 than the article model (0.86 vs 0.67 f1 scores).
After more systematic data inspection, we have tuned our generators (`realistic_experiment_articlemodel_local`) to produce more local
dependencies, which made the article model perform better - 0.81 f1 score, which still did not beat the baseline.

To summarize:

We have made an experimental environment, that allows for quick experiment setup.
We tried to emulate invoice documents based on assumptions on the data
(observation about the data being close todo add figure)
(assumption that the result box class depends only on some boxes --- comes from the fact that amount total: 100$ is fairly simple)
and so our simulated setting does not match the original environment only in one aspect - 
the number of seemingly unimportant text boxes to the human eye.

We have created a problem, that seemed similar to invoices and easy to get to 0.86 score with a simple baseline.
But in reality, all the models, that are stronger on business documents failed and scored below the baseline.
Therefore what are the key takeaways from our modelling experiments? 

The created problem of simulated business documents is indeed harder or at least different - our assumption about unimportant information was not valid.
Therefore the models that are stronger on business documents are exploiting structures, similarities and bonds concerning the boxes,
 that are not easily visible for the human eye.
That might mean, it could see some commonalities in the layouts logic in the real data (which we were not simulating, since our simulations are layout free).
Note we do not use layout information in our models anyway, the networks see all various types of documents as they are,
and yet it seems that they are able to use commonalities to get to better results, since when we generate data without any layouts, the results get worse. 


To make a fast label reuse experimental environment, we need to abandon the idea of simple generators and proceed with dropping the visual information as that is the most memory heavy one.

 
