# Introduction

Verification: verifying the correctness of the machine learning's expected behavior
Validation: validating whether the machine learning software do what it claims to be able to do.

## Challenges

### Can humans understand how ML work?

Machine learning "learns" from training data, in which results can be intuitively interpreted as weighted combination
of "features". Many machine learning systems are very complex, in other words, such as weighting is inscrutible comming from
these machine learning systems, or at least not intuitive. As a result, there is an unknown (significant?) chances
that results are brittle. For example, due to accidental correlation in trainning data, or sensitivity to noise.

This can be very dangerous for having ML perform life-critical tasks, e.g., driving a fully autonomous car.

Many ML approaches have random number generators, in unit test case, if one carefully control the random number generator
it may be possible to reproduce all behaviors. However, such ML is generally sensitive to initial conditions at system
level. which implies that it can be essentially impossible to get test reproducibility in real systems. In fact,
significant effort in many real-world ML systems are used to force or "trick" ML into displaying desired behavior.


### Oracle Problem
It is difficult to detect faults in machine learning applications because often there is no "test oracle" to verify the 
correctness of the computed outputs.
 
In many supervised/semi-supervised/reinforcement learning approaches, algorithms "learn" knowledge from "test data". For
a particular learner whose specification is fixed, the prediction result must be deterministic. However, the prediction 
always involves very complicated logical and computational process, and brings difficulties in figuring out the expected
result, for any arbitrary training data and test data, unless we can repeat the whole process with a "perfect" version 
of the program. Obviously such a "perfect" version never exists in the real-world. This makes the learners fall into 
the category of programs having the oracle problem.

Many data analysis tools are complex and are difficult to be tested due to the absence of test oracles.

### Training Data

Usually the testers for software implementing machine learning just utilize some special test cases which were
acquired from previous studies or domain knowledge, and seldomly conduct comprehensive testing. Such an approach is
unsatisfactory because these softwares usually serve as the kernel and functional components in many applications.

### Data Selection and Validation

Data sleection and validation are critical to the effectiveness and performance of big data analysis, but large volume
and varieties of big data create a grand challenges for the selection and validation of big data. Existing work
has shown that abnormal data existing in datasets could substantially decrease the performance of data analysis 
softwares.

Furthermore, there may not be a way to know that the training data is "completed". for life-critical system, the training
data must be safety critical and people's perception of "almost the same" does not necessarily predict ML responses.


### Big Data Integration with Machine Learning

The machine learning algorithms used for processing big data are difficult to be validated given the volume of data
and unknown expected results. Although there are significant work on the quality assurance of big data, verification
and validation of machine learning algorithms and "non-testable" scientific softwares, few work has been done on 
systematic validation and verification of a big data system as a whole.

At the system level, a big data system can be verified and validated using regular software verification and validation
techniques. However, regular system verification and validation approaches are not good enough for ensuring the quality
of the essential components in a big data system: big data, machine learning and "non-testable" software components.

# Approaches

## Synthetic Testing versus Real-time Monitoring

Majority of testing is done in the synthetic testing in which prepared data are fed to machine learning system to verify
and validate. 

Real-time monitoring is more appropriate during the stage of "software maintenance" in which the data analytics softwares
have been deployed and its behavior need to be monitor for correctness verification and quality validation. For example,
for fully autononmous vehicle verification and validation, run-time safety monitors using traditional high-ASIL softwares
can be used. 

# Approaches in Synthetic Testing

* metamorphic testing: metamorphic testing does not directly verify whether the correctness of each individual testing results; instead it 
checks whether they satisfy certain expected properties
* mutation analysis: insert syntatically correct mutant (e.g. configuration of parameters or changing operators)
* cross-validation analysis: detect anomaly by validating the performance consistency
* Selection and validation technique for big data machine learning: validation and automatic selection of big data for
machine learning softwares as well as verification and validation of machine learning software on big data.
* Machine learning-based approach for verification and validation: data analytics softwares can be tested using machine
learning-based approach.
* Accelerated Stress testing via fault injection
* Robust Testing

## Metamorphic Testing

Metamorphic testing aims to detect faults in machine learning softwares without test oracles. 

### Principle

Metamorphic testing uses properties of functions such that it is possible to predict expected changes to the output for
particular changes to the input. Although the correct output cannot be known in advance, if the change is not as 
expected, then a fault must exist.

### Method

In the metamorphic testing, first enumerate the metamorphic relations (MRs) that machine learning would be expected to 
demonstrate, then for a given machine learning implementation determine whether each relation is a necessary property
 for the corresponding algorithm. 
* If yes, then failure to exhibit the relation indicates a fault (verification on correctness)
* If the relation is not a necessary property, then a deviation from the "expected" behavior has been found (validation)

For many algorithms with stochastic behavior, statistical inference methods can be used to determine 
the verification and validation.

### Examples of Metamorphic Relations

* Consistence with affine transform: the prediction result should be the same if we apply the same arbitrary affine 
transformation function to the values of any subsets of attributes for each sample in the training data set and test 
data set.
* Permutation of class labels: for a class label permutation function Perm(), there should be one-to-one mapping 
between a class label to another label. 
* Permutation of the attribute: the prediction output should remain unchange if we permutate the m attributes of all 
the training and testing data.
* Addition of uninformative attributes: the prediction output should be the same if we add uninformative attributes to
be associated with each class label.
* Addition of informative attributes: 
* Consistence with re-prediction: the prediction output should remain unchanged if we append, for example, testing data
back into the training data.
* Addition of classes by duplicating samples: prediction output should remain unchange if we duplicate classes or/and
samples
* Addition of classes by re-labelling samples: prediction output should be the same for this case 
* Removal of classes: prediction output should remain unchanged for this case
* Removal of samples: prediction output should remain unchanged for this case.


## Mutation Analysis

Mutation analysis is used for evaluation of effectivenss of machine learning softwares.

In mutation analysis, faults are systematically inserted into the machine learning application of interest. 

Mutants are used to emulate real faults and serves as good proxy for comparisons of testing techniques.

### Method

In mutation analysis, mutants can be systematically generated using another softare (e.g., MuJava). The source files can
be mutated by methods such as:

* method-level operators 
* class-level operators

Such automatic mutation analysis system generates many syntatically correct mutatants using method-level and class-level
operators.

### Examples of method-level operators

* AOR: Arithmetic Operator Replacement
* ROR Relational Operator Replacement
* COR Conditional Operator Replacement
...
* ASR: Short-Cut Assignment Operator Replacement

## Cross-validation Analysis

Cross-validation analysis can be used to detect anomaly in machine learning softwares by checking the performance 
consistency of the machine learning softwares over cross-validation strategies.

Many different scores can be used for cross-validation analysis such as:

* Classification Problem: Precision, Recall, Fallout, F1 scores, AUC under ROC, Information Scores, Multi-criteria measurement
* Regression Problem: RMSE, R^2, p-value, AIC, BIC

These are coupled with statistical testing for hypothesis testing.

## Verification and Validation of Big Data Machine Learning System.

This technique aims to verify and validate big data machine learning system.

A proposed framework as given includes task in three layers:

* the foundation layer is technique for automated selection and validation of big data
* the middle layer is an approach for verification and validation of the machine learning algorithms including feature
representation, extraction and optimization
* the top layer is an approach for testing domain modelling systems, data analytics tools and applications.

The automatic selection and validation of big data can be done in the following steps:

* feature optimization for data classification
* feature optimization for data selection
* domain-specific proccessing based data selection
* deep learning based data selection

The verification and validation of machine learning and "non-testable" system parts can be done via metamorphic testing
and iterative metamorphic testing.

## Machine Learning approach to Data Analytics Software Testing

Machine learning can be used to build software testing automation for data analytics softwares. Such an approach can 
assist developers to improve their test strategy in verification and validation of a data analytics software 
systems. 

### Testing approach

The testing approach can be black-box, white-box, and gray-box. The black box approach, testing can be perfomed using 
 external description of the software system such as the software specification. In a white-box approach, the internal properties of the data analytics software system like source code can be used for testing purposes. The grey-box approach is a combination
of the two.

For example, black box testing can be done using Category-Partition for test specifications and test suites.
 
In white-box, fault prediction can be carried out using data extracted from source code using classifiers.

### Testing general activity

ML can be used to automate different sub-domains in the SDLC:

* test planning
* test case management
* debugging 

For example, in the test planning sub-dimension, <i>testing cost estimation</i> can help test managers to predict 
testing process cost and time and provide good testing plan to manage the testing process.

ML in the test case management sub-domain can be used to proritize test case,design test case, refine test case.

In the debugging sub-domain, ML can be used to prioritize bug priority, fault localization, etc.

### Learning property

Various types of data can be used to build the target system. The training data can be collected in different stages
of the software testing processing for software development life-cycle. The learning elements could be software metrics,
software specification, CFG (control flow graph), call graph, test case, execution data, failure reports, and/or coverage
data.

## Stress Testing and Robustness Testing

Traditional approaches for testing on critical systems can also be used to test data analytics softwares.

Example of robust testing:

* detect system failures due to improper handling of floating-point numbers in ML systems
* detect system failures due to array indexing and allocation
* detect system failures due to time flowing backwards, jumps
* detect problems handling dynamic states

Testing philosophy should include black swan events

# Reference paper

* A Machine Learning Based Framework for Verification and Validation of Massive Scale Image Data
* Testing and Validating Machine Learning Classifiers by Metamorphic Testing
* Machine Learning-based Software Testing: Towards a Classiﬁcation Framework
* The Application of Machine Learning Methods in Software Verification and Validation 
* Performance evaluation for learning algorithms
* Challenges in autonomous vehicle testing and validation
* A Bayesian Metric for Evaluating Machine Learning Algorithms





