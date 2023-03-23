# Pet Dataset Segmentation
Final project for machine learning and data mining

**Team Members:**
- Vedanshi Shah
- Vedant Bhagat
- Sid Mallareddygari

## Dataset [[Here](https://www.robots.ox.ac.uk/~vgg/data/pets/)]
## Project working document [[Here](https://docs.google.com/document/d/1e44iOMHAqkH2ctta_33C5Q0RKm1GWLZg1SSbtU3Vgq4/edit?usp=sharing)]

## Project summary
To be able to recognise the pets (between cats and dogs) and their breeds based on the data given.

## Documentation
1. Introduction: Summarize your project report in several paragraphs.
    1. What is the problem? For example, what are you trying to solve? Describe the motivation.
    2. Why is this problem interesting? Is this problem helping us solve a bigger task in some way? Where would we find use cases for this problem?
    3. What is the approach you propose to tackle the problem? What approaches make sense for this problem? Would they work well or not? Feel free to speculate here based on what we taught in class.
    4. Why is the approach a good approach compared with other competing methods? For example, did you find any reference for solving this problem previously? If there are, how does your approach differ from theirs?
    5. What are the key components of my approach and results? Also, include any specific limitations.
2. Setup: Set up the stage for your experimental results.
    1. Describe the dataset, including its basic statistics.
    2. Describe the experimental setup, including what models you are going to run, what parameters you plan to use, and what computing environment you will execute on.
    3. Describe the problem setup (e.g., for neural networks, describe the network structure that you are going to use in the experiments).
3. Results: Describe the results from your experiments.
    1. Main results: Describe the main experimental results you have; this is where you highlight the most interesting findings.
    2. Supplementary results: Describe the parameter choices you have made while running the experiments. This part goes into justifying those choices.
4. Discussion: Discuss the results obtained above. If your results are very good, see if you could compare them with some existing approaches that you could find online. If your results are not as good as you had hoped for, make a good-faith diagnosis about what the problem is.
5. Conclusion: In several sentences, summarize what you have done in this project.
6. References: Put any links, papers, blog posts, or GitHub repositories that you have borrowed from/found useful here.

## Project proposal [[Here](https://www.canva.com/design/DAFd3gYxx48/vUWV_t2ObbldzB7A1JCsnw/edit?utm_content=DAFd3gYxx48&utm_campaign=designshare&utm_medium=link2&utm_source=sharebutton)]
  - To be able to recognise the pets (between cats and dogs) and their breeds based on the data given.
  - Interesting because:
      - Large and Diverse
      - Challenging
      - Fine-grained Recognition
      - Benchmark
  - Approaches:
      - Since the input data is images, neural networks seems to be the most apt method for classification
      - Specifically Convolutional Neural networks (CNNs) may work best due to the large number of pixels, i.e. data, in the inputs
      - filtering down to reduce the computation is very important and CNNs are able to scale to handle large amounts of data
      - CNNs can also focus on details that are important for classification, and will not get lost due to a large amount of data
      - However, based on the information from the website, we can see that they recommend the use of segmentation and that might be an interesting approach to take
    - Timeline:
      - We aim to finish the data cleaning, preprocessing, repository and model set up in one week
      - Then we will work on creating the CNN and/or segmentation model
      - Then we would have preliminary results by April 12th, in preparation of our presentation sharing our project results
      - This leaves us another two weeks to focus on model validation and fine tuning our results based on feedback from our presentation for the final deadline of April 27th

