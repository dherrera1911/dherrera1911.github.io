---
title: 'Is the world as we see it?'
date: 2024-08-12
permalink: /posts/2024/08/is-the-world/
tags:
  - Perception
  - Biology
  - Philosophy
  - General audience
---

Author translation of the chapter "Es el mundo como lo vemos?"
([link to pdf](/files/teaching/hym/capitulo_hym_dh.pdf) in Spanish)
for the book
["Hitos y Mitos del Cerebro"](/home/dherresp-adm/Docencia/Hitos_mitos/diapositivas_2021/imagenes/cueva_platon.jpeg), edited
for the homonymous first year course at
Universidad de la República, Uruguay. The chapter aims
to introduce several ideas of perception and neuroscience to
first year students through the guiding question of whether the
world is as we see it. In particular, the chapter discusses
the question from three perspectives: an evolutionary perspective,
a resource optimization perspective, and the perspective of
ambiguity of sensory information.

Introduction
------

Approximately 2,400 years ago, Plato’s Republic was published
in ancient Greece, containing his famous allegory of the
cave (**Figure 1**). This
allegory describes a group of people living inside a dark cave, where
the only visible things are the shadows projected from the entrance of the
cave onto one of its walls. Since they cannot see anything else,
these people think that the world is made up of shadows and
cannot conceive the world as we perceive it. This allegory makes us
question the possibility that we might be in a situation similar to these
people, and that our perception presents us with a distorted world.
Thus, Plato raises the central question of this chapter: is the world
as we see it? Despite its long history, philosophy has not
abandoned this question, which, with its variants and other colorful examples
like Descartes' demon from 1641, or Putnam's brains in vats has
been one of the main guiding questions in the history of philosophy.


<figure>
  <img src="/files/blog/hym/cueva_platon.jpeg" alt="Plato" title="Plato" width="50%">
   <figcaption> <strong>Figure 1:</strong> Plato's cave allegory </figcaption>
</figure>

Although this question might seem simple,
it can be interpreted and analyzed
in many different ways. Therefore, to answer it, we
must clarify what we mean by the words "see" and 
"world". For example, if we take "see" as everything
we can know with our senses and through science, are
asking whether there is a "world" that escapes us and that
we cannot know by any method (or if we live in some
sort of cave from which we cannot escape). This question
would fall within the field of philosophy. On the other hand,
if we take "see" as what we can perceive with our
senses and "world" as the physical world,
then we are faced with a question for physics. This question has
a clear answer: the world is not as we see it.
There are subatomic particles, dark matter, magnetic fields,
and many other physical entities that escape our sensory organs.

In contrast to those other approaches, in this chapter,
we will analyze the question from a biological and neuroscientific
point of view. The version of the question we will study
can then be posed as follows: is what we see with our senses
an accurate image of the world at our physical scale?

What is seeing? Visual system general overview
------

Before we begin to analyze what biology says, it is necessary to
reflect on what it is to *see*. Because our visual system
provides us with an excellent ability to see so
effortlessly, it difficult for us
to appreciate the complexity of the task. To gain intuition about how
complicated seeing is, let us consider a simple system: a robot
with vision.

Suppose we have a small robot, and we
want to make it *see*. The first thing that we need to
do is give it a detector that receives light, or the equivalent
of an eye. This could be simply a digital video camera.
This camera receives the image of the scene and
transforms it into a matrix of
numbers, indicating the intensity of light at each point.
This matrix of numbers constitutes a digital image,
where each number represents a pixel (**Figure 2**).

But just taking pictures is not enough.
Seeing involves extracting relevant information about the world
from the pictures (e.g., what object is in front of me; how far away is it;
what size is it, etc.). To do this, we must give the robot
a computer with a program that extracts this information from
these matrices of numbers.
It is not difficult to realize how complex such a program
needs to be to allow the robot to convert those
number matrices into assessments like:
"I have a medium-height marble table two meters in front
of me." For example, note that the numerical matrix (i.e., the raw image)
can change radically if we modify the light source, move the robot
slightly, change the background of the image, or rotate the table.
The program faces the difficulty
of giving the same result despite all those "trivial" changes.
These difficulties can be seen in how difficult it is
to develop fully self-driving cars despite the immense
interest in this technology.
Science and engineering have not yet solved this problem that
our brain solves so effortlessly.


<figure>
  <img src="/files/blog/hym/imA.png" alt="Foxy" title="Computation vision" width="70%">
   <figcaption> <strong>Figure 2:</strong> A digital image can be thought of as a matrix of numbers. On the left, we see a black-and-white image and an enlargement of a segment of it. On the right, we see the same image segment, but with numbers between 0 and 100 indicating the intensity value of each pixel. A computer program that "sees" must take that matrix of numbers and extract information from the scene (e.g., that it contains a fox) from it.</figcaption>
</figure>

What is interesting about the robot example is that our visual system is
not fundamentally different. Light enters our eyes and forms an image on
the retina (located at the back of the eye). The retina
is covered with light-sensitive cells called photoreceptors. Photoreceptors are active
when they do not receive light and become inactive when illuminated. In
this way, a photoreceptor with greater activation indicates a dark region of
the image, and one with less activation indicates a lighter region,
thus representing the image through neural activity. This system, by which
the degree of illumination is sensed, can then be thought of as
the one described earlier, with each photoreceptor acting like an individual pixel
and its degree of activation as the numerical value in the matrix that
is the image.

Then, just like the robot, the
image must be processed to extract the relevant information. In our visual
system, after initial processing in the retina, the optic nerve transmits
visual information to the brain, where we have dozens of cortical areas
dedicated to extracting information about the world. These areas are densely
interconnected, and their functions are still not fully understood, but they carry
out the processing that allows us to recognize a face, estimate a
distance to throw a projectile, or choose the apple we like the
most from drawer. This processing occurs subconsciously, and we do
not have access to it, but it is important to remember that
"seeing" constitutes a complex processing and interpretation of the image by
the brain.

Having established the complexity behind our ability to see,
we are in a position to begin answering the question, that
is, whether the processing our brain does of the image results in
a faithful description of the world around us. To do this,
we will analyze the question from three perspectives: 1) the
evolutionary perspective; 2) the cost of processing information perspective,
and 3) the ambiguity in the data perspective.


Evolutionary Perspective: The Visual System Did Not Evolve to See the World as It Is
------

Like the rest of the systems and biological processes that make up our
organisms, the visual system evolved to contribute to our reproductive success.
That is, evolution did not necessarily select for the visual system that best
represents the environment, but rather the one that most favored survival and
reproduction. An example that illustrates this point is our color vision,
which went through an interesting path to reach its current state.


<figure>
  <img src="/files/blog/hym/imB-pre.jpg" alt="Waves" title="Color wavelength" width="50%">
   <figcaption> <strong>Figure 3:</strong> Color vision corresponds to
  the ability to distinguish the wavelength of light. Image 
  taken from
  <a href="https://www.sciencelearn.org.nz/resources/47-colours-of-light">https://www.sciencelearn.org.nz/resources/47-colours-of-light</a>.</figcaption>
</figure>


Light is an electromagnetic wave, and like other waves, it is
characterized in part by its wavelength (**Figure 3**).
Color vision is the ability to
distinguish the wavelength of the light we perceive. How is this achieved?
To begin with, a particular photoreceptor has a preference for certain
wavelengths that can more easily modify its activation (**Figure 4**).
But despite this preference, a single photoreceptor does not allow us
to distinguish between different wavelengths, because its
activation also depends on the intensity of the light.
This means that a given photoreceptor activation
can represent a high intensity of a non-preferred wavelength,
or a low intensity of a preferred wavelength.
On the other hand, two photoreceptors with
different wavelength preferences do allow us to discriminate between
wavelengths, letting us separate the contribution
of light intensity and wavelength. Having more
photoreceptors with different preferences allows an
organism to better discriminate wavelengths, improving color vision.


<figure>
  <img src="/files/blog/hym/imB.jpg" alt="Waves" title="Photoreceptors" width="70%">
   <figcaption> <strong>Figure 4:</strong> 
  Humans have three different photoreceptors for color vision, called cones, and each has its preference for different wavelengths. The graph shows the ability of each cone (named blue, green, and red) to absorb different wavelengths (whose color is indicated in the bar below). A single cone does not allow for disambiguation of wavelength because its activation depends on the color of the light and its intensity (e.g., it cannot determine if greater activation is due to a change in light color or a change in its intensity).</figcaption>
</figure>


At one point in evolution, the subphylum of vertebrates came to
have four different types of color photoreceptors.
But then at another point in our evolution,
mammals lost two of these photoreceptors (this was when we
were nocturnal, and color vision was not as important for us),
leaving them with only (or dichromatic vision). At a later point,
however, primates acquired a new photoreceptor,
bringing our total to three (or trichromatic color vision).
This photoreceptor falls in the middle of our visible color
spectrum, and makes
it easier to discriminate green from other colors. The main hypothesis about
what led this photoreceptor to be selected is that it
was particularly favorable for better perceiving vegetation,
for example, the fruits that were part of our diet against the
green background of trees.

These two evolutionary events, the loss and re-gain of photoreceptors,
show how evolution can select for improvements in sensory systems
representation of the world, but that it can also lead to
its deterioration. In this way, evolution has
led us to currently perceive fewer colors than many species
of birds that still maintain the four original
vertebrate photoreceptors.

<figure>
  <img src="/files/blog/hym/imB-post.jpg" alt="Waves" title="Color evolution" width="50%">
   <figcaption> <strong>Figure 5:</strong> Two color photoreceptors
  were lost by mammals in evolution, making our color vision worse.
  We humans acquired a new photoreceptor making us thrichromats, as
  opposed to most other mammals. Image by Jen Christiansen, taken
  from "What Birds See" by Timothy H. Goldsmith, Scientific American
  July 2006.</figcaption>
</figure>


Another interesting example of the relationship
between evolution and perception is the study of the frog's visual
system presented in the article
["What the Frog's Eye Tells the Frog's Brain"](https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=4065609) by Lettvin
et al., published in 1959. The experiment
carried out in this article involves showing visual stimuli to a frog while
recording the activity of the fibers of its optic nerve (which take
the information from the eye to the brain) to determine
what information about the
scene these fibers transmit to the brain. One interesting
thing about this article is that the authors discuss the
results in the context of the
natural behavior of the specimen, which uses vision to hunt and escape
predators. For example, the frog's visual behavior has the
peculiarity that it seems not to perceive the static elements of the world
around it (e.g., food does not catch its attention
if it does not move, [see video](https://www.youtube.com/watch?v=boytEUqImMI&t=25s)).
Moreover, its predatory behavior is mainly
guided by the movement and size of visual objects: it will try
to capture any small object that moves like an insect, even if
it looks very different from an insect to us.

The authors discuss that this behavior fits
very well with their findings on fiber activation.
They describe four types of fibers in the optic
nerve, each reporting about very specific visual patterns in the
environment. For example, one of the types of fibers described
by the authors is activated when a
small shadow stops at a specific point in the visual field
and moves intermittently.
This type of fiber seems to correspond to the behavior of the
frog described above, responding to the movement of small objects
that move like insects. Therefore, the brain receives
highly pre-processed information 
where a large part of the visual world detail is discarded,
which may help explain why the frog's visual behavior is
so limited. The frog will then be able to
detect this type of visual event, but not others for which it
does not have fibers.

But, although the frog's
perception may seem limited to us, what is important is that it
allows it to generate the necessary behaviors to survive and reproduce. For
example, concerning the fiber described earlier, the authors ask: "*can
one describe a better system for detecting an accessible insect?*" So,
although the frog's visual world is limited (for it is
composed of some simple patterns like moving dots), it is sufficient for
its ecological needs. Although it is easy to fool a frog in
the laboratory with something that looks like an insect, the relevant
question is: how many objects that are not insects have that size and
move like them in its natural environment?

It is possible to notice a similarity between the frog case
and the allegory of the cave, and although our visual system
is much more complex, it is natural to ask now: to what
extent do we suffer from limitations as great as those of the frog?
In a certain way, our visual system carries out a similar
process: the visual stimulus is already processed even in the retina,
and in each visual area of the cerebral cortex, specific stimulus patterns
are extracted and passed on to higher areas, and those were
selected because of their contribution to our reproductive
success. In this process, information about the world is
also lost. For example, if we look at two different
images of white noise (the static present in old televisions,
**Figure 6**),
these images are very different considered pixel by pixel, but
they look identical to our visual system, which discards
a lot of this information.

<figure>
  <img src="/files/blog/hym/noise1.png" alt="Noise" title="Noise" width="50%">
   <figcaption> <strong>Figure 6:</strong> Two different
    white noise images, that although they are very different pixel-by-pixel,
    look identical to our visual system.</figcaption>
</figure>


However, there are
reasons to think that our visual system is not as affected as that
of the frog. For example, the patterns detected in the early
stages of our visual system (point-like structures in the retina;
stripe-like structures in the primary visual cortex) are the
same as those reached by mathematical tools that seek the best method of
representing images. This suggests that perhaps we are capturing a large part
of the information available in images.
This observation also aligns with the argument that
flexible goals and behavior such as that in higher mammals
requires faithful representation of the world (e.g. see
a technical analysis [here](https://onlinelibrary.wiley.com/doi/full/10.1111/cogs.13195)).
Thus, it could be expected that
evolution has led us to see the world approximately as it is.

In summary, our visual system has been shaped by evolution
to perceive those aspects that are important for our survival.
However, our complex and adaptable behavior would seem
to require the ability to faithfully perceive our environment, and some
computational studies support this reasoning. Although there are aspects of
the world that we do not perceive, the evolutionary perspective leaves open
the possibility that we perceive the world accurately.


Perspective on the Cost of Representation: Representing the World Faithfully Is Expensive
------

Our brain is finite and has limited resources. Capturing and processing visual
stimuli is costly, so our ability to perceive the world faithfully is
also limited. However, evolution tends to lead to the efficient use
of resources, and it happens that our visual system manages them
in very efficient ways, allowing us to perceive correctly what is relevant to us.
An important and illustrative aspect of resource use by our visual system is the
separation between central and peripheral vision, which we describe below.

It is evident that vision plays a fundamental role in our cognition,
which can be expressed by saying that we are a "visual animal."
In line with this, the brain dedicates a significant amount of
resources to visual processing. But these resources are not distributed evenly to
process the entire visual field; instead, their distribution marks a very
strong distinction between two components of vision: central vision
(corresponding to the part of the retina called fovea) and
peripheral vision. These components are evident to us through introspection:
while we perceive very clearly the part of the visual scene on
which we focus our gaze, what falls in our peripheral vision is much less clear.
An obvious example of this is that it is impossible for us to
perform some tasks (like reading) with peripheral vision.


<figure>
  <img src="/files/blog/hym/imC.png" alt="fovea" title="Fovea" width="50%">
   <figcaption> <strong>Figure 7:</strong> 
  The resources of the visual system are not distributed homogeneously across our entire visual field. The image shows a map between the visual field (on the left) and the primary visual cortex (on the right). We see that nearly half of the primary visual cortex is dedicated to processing a small fraction of the visual field called the fovea.
  </figcaption>
</figure>


This division between central and peripheral vision is so natural to us that we
don't usually question it, but why do we see with
little clarity in the periphery?; would it be possible to have the
high resolution of central vision across the entire visual field? A key
to understanding this is that although our central vision occupies a small fraction
of the visual field (we can roughly define it as the visual area
covered by our closed fist with an extended arm; **Figure 7**), it
uses approximately half of the resources of our visual system.
Thus, although peripheral vision occupies the majority of the visual field,
it uses only the other half of the cortical resources. If
we extrapolate the relationship between visual field and cortical area from central
vision, processing our entire visual field with the sharpness of central vision would
require a much larger brain than we have, which seems biologically
impossible.

But even though we clearly see
only a small part of the visual field, we use an important
tool to make the most of the resources available: eye movements.
We are constantly moving our eyes (and head) to capture the
important elements of the environment with high precision. This scanning of the
scene is something we know how to do so well that it feels
natural and simple, and we don't notice our constant eye
movements. But our eyes are always subconsciously choosing
for the most relevant aspects of the scenes to
scan, allowing us to obtain large amounts of information with limited resources.
With this efficient allocation of resources, we manage to attain
a feeling of a detailed perception. But despite the efficiency of the system,
we often overestimate our perception, as the following examples illustrate.

The phenomenon called change blindness is demonstrated experimentally
by flashing in an alternating fashion two identical images
that differ in some specific element
([see YouTube example](https://www.youtube.com/watch?v=FWVDi4aKC-M)).
The experiment consists of the participant identifying
what the differences between the images are. What is interesting about the
phenomenon is that even large differences between the images are
difficult to identify. This result contrasts with our subjective impression:
although we believe we perceive the image clearly,
the reality is that it is difficult for
us to remember even its most conspicuous elements.

Another example is inattentional blindness. This phenomenon
consists of certain elements of images to which
we are not paying attention going unnoticed, even though they are very
visible. An iconic example
([see here](https://www.youtube.com/watch?v=UfA3ivLK_tE) before
you continue reading) consists
of a video showing a group of people passing a basketball, and
the task is to count how many passes are made. At the
end of the video, it is revealed that a curious animal made
an undisguised appearance that goes unnoticed to the viewer
concentrated on counting the passes.

Another example, striking because of the participants expertise,
is an experiment in which radiologists were asked to make
a diagnosis on some tomography plates in which a visible image of a
gorilla had been included (**Figure 8**). Although the radiologists
examined the image closely, and almost all laid their eyes on the
gorilla (as measured by the eye-tracking technique), when asked
at the end of the experiment, most had not noticed the hidden
ape. This last example also shows that it is not enough to
have something in central vision to process it correctly. The allocation of
resources in the visual system does not only occur in eye movements but
also at the level of information processing, prioritizing one processing "pathway"
over another.


<figure>
  <img src="/files/blog/hym/imD.png" alt="Gorilla" title="Gorilla" width="50%">
   <figcaption> <strong>Figure 8:</strong> 
  Inattentional blindness is a phenomenon where aspects of the visual field that we do not expect or are not paying attention to go unnoticed. The image shows a classic experiment where a gorilla was inserted into tomography plates, as shown in the enlargement. In the experiment, radiologists were asked to evaluate the tomography without being informed of the gorilla's presence. Most radiologists did not detect the presence of the gorilla, despite many of them fixing their gaze on it (the white circles on the right show the eye movements of one of the radiologists). Image taken with permission from 
  <a href="https://journals.sagepub.com/doi/full/10.1177/0956797613479386?">Drew, Vo & Wolfe (2013) Psychol. Sci. 9. 1848.</a>
  </figcaption>
</figure>


These examples illustrate how costly it is to
process visual stimuli and how, at a given moment, we perceive
only a small part of the scene with clarity. But is this
something new? Before starting this section, any of us already knew
that there are elements of the scene that we cannot perceive at a
given moment, for example, what is behind us. Although it
is interesting how our "real" or "objective" perception seems
worse than we believe, this new discovery does not seem to close
the issue since, from the beginning, we know that our visual
system has limitations.

Moreover, these limitations do not imply that we
cannot perceive well the specific elements of the world on which we focus
our attention, just as not having eyes on the back of our
heads does not mean we cannot turn around to see what is behind
us. And although the examples discussed are striking, it is important
to note that the real world is very different from the experimental manipulations
that show these effects (e.g., in the real world, objects do not
suddenly disappear as in change blindness experiments). From
this, the next natural question is: the part of the scene
on which we focus our attention and which we observe closely, is
it as we see it?


Perspective on the Ambiguity of the Data: Seeing Requires "Interpreting" the World
------

The last aspect that we will consider 
is a fundamental aspect of the computational tasks of
vision: visual stimuli are ambiguous, so we must interpret
them to obtain information about the world.
This stems in part from the fact that
our visual input is two-dimensional (the plane of our retina, like the
two dimensions of a photograph), while the external world
has three spatial dimensions. Our visual system is very good at making
(subconscious) interpretations of visual stimuli, making it difficult for us
how they can be ambiguous, but nonetheless these necessary
interpretations occur at various stages of visual processing.

<figure>
  <img src="/files/blog/hym/imE.jpeg" alt="Daltmatian" title="Daltmatian" width="50%">
   <figcaption> <strong>Figure 9:</strong> 
  This figure shows a classic image of a scene that has been converted to black and white, resulting in the appearance of a set of scattered black spots. But despite the chaotic image, our brain interprets this set of spots to see the three-dimensional scene of a Dalmatian walking towards a tree.
  </figcaption>
</figure>


For example, one of the first steps in visual processing
is to group the pixels into segments or surfaces (a process we call 
image segmentation"). This involves using some criterion to determine
when two regions of the image belong to the same surface. But this is not
easy. We can think of some simple rules to group image
sections together (e.g., group pixels that have similar color and
are close), but these simple rules will quickly fail in many
real world cases. Our brain manages to group the elements
of the images using complex rules that we still do not fully
understand. For example, **Figure 9** shows a classic scene
with black spots spread across an image with very coarse
information. But despite the chaotic appearance of the image, our brain
manages to group the dots together into surfaces, group these into
objects, and finally into a scene. Looking at how chaotic
the image is, it is difficult to understand what evidence our
brain uses to interpret that chaos of pixels.
But beyond the details of processing, what matters here is that
our brain is having to interpret many aspects of the image
(such as what areas of the image are from the same surface),
and that in general many interpretations are possible and
our brain must choose one. The next example illustrates
how this ambiguity in possible interpretations can lead to
illusions.

<figure>
  <img src="/files/blog/hym/imE2.jpg" alt="Ames" title="Ames" width="50%">
   <figcaption> <strong>Figure 10:</strong> 
  The Ames room is a famous illusion that can occur in the real world, and
  is used in science shows and fairs. Using a room with irregular
  geometry, the illusion makes a person look tiny or giant, depending
  on their position in the room.
  </figcaption>
</figure>


The Ames room is a classic illusion in which we generate a misinterpretation of
a scene. This is a room with a particular shape: it
is not rectangular, and both its walls and ceiling and floor are
sloped and not parallel (**Figure 10**). To create the illusion, one of its
walls has a carefully placed hole through which we can look and perceive
a normal rectangular room (with parallel walls, ceiling, and floor).
When we look through the hole, our perception is wrong,
as we believe we see a normal room when, in fact,
we have a distorted room in front of us. Moreover, if
there are people in the room, depending on their position, we
will perceive them as giants or as tiny
(see a video [here](https://www.zmescience.com/feature-post/health/mind-brain/optical-illusion-ames-room/)). The important thing to
note here is that the visual stimulus in this case (and in
all others) is ambiguous. It is compatible with both a rectangular
room (as we mistakenly perceive) and the actual scene of the
irregular room (since this is, in fact, what generates the
image). In fact, the visual stimulus is compatible with infinite possible
scenes. For example, it could be a giant room with giant
people, a small room with small people, or it could have
many different shapes with people of various sizes. The image is compatible
with all these alternatives because all of them could generate the same pattern
of light in our retinas, and interpreting the image involves choosing one
of these alternatives.


This example show that our visual system makes interpretations of visual
stimuli to perceive the scene around us, and that
these interpretations can be wrong. But how does our brain
arrive at these specific interpretations? Although this question is
still an area of research, neuroscience can give us some answers. First, it
is important to clarify that to investigate this, we normally study simplified
stimuli that aim to understand the processing of a very specific type
of information. Some examples could be how we use color to segment
images or how we use visual textures to estimate distances. Despite the
gap between these simplified experiments and our "natural" perception, they
allow us to draw two relevant conclusions to our question: 1)
the visual system uses a set of "rules" that allow us
to construct the interpretation of the image, and 2) these rules
are not arbitrary but are based on the structure of the world we
inhabit.

In the process of choosing which of the possible scenes
is the one that generates the image, the brain imposes restrictions 
(or rules) on the interpretation that helps select some possibilities and discard
others. These rules mark how to interpret certain elements of the image
and are applied at several levels and in parallel. For example,
a phenomenon observed with simple stimuli is that our vision groups the elements
of the image that are aligned in the direction of their orientation,
that is, they are are collinear (**Figure 11**). On the contrary,
elements that are aligned in the direction perpendicular to their orientation do not
tend to be grouped. Then, using this rule to interpret natural
images, our visual system can choose the interpretation of the scene that
groups the elements that are collinear and discard other interpretations.

<figure>
  <img src="/files/blog/hym/imF.png" alt="Gabor" title="Gabor" width="50%">
   <figcaption> <strong>Figure 11:</strong> 
  The grouping of image elements is a fundamental process in visual perception, occurring according to complex rules that we still do not fully understand. The image on the left shows an example of grouping, where our visual system groups the collinear elements despite being surrounded by other similar elements. In the image on the right, we have the same elements but orthogonal to the alignment direction, and our visual system does not group them.
  </figcaption>
</figure>


Another rule that the visual system uses is that it tends to assume that the
light source comes from above. A common ambiguity is that the same
image can be generated by a concave surface illuminated from above and by
a convex surface illuminated from below, and vice versa. This ambiguity
and our brains preference for light from above is why
when seeing an image of a crater upside down, it looks like a
hill (**Figure 12**).
In the face of this type of ambiguity, our
visual system will tend to choose the scene where the light comes from
above and then interpret whether the surface is concave or convex according to
this criterion. Thus, by applying a large number of rules,
the visual system can choose one interpretation over another at different
levels until reaching a global interpretation of the image we receive, which
will be our perception. But where do these rules come from?; how do
we know if they are good for perceiving the world?


<figure>
  <img src="/files/blog/hym/crater.png" alt="Gabor" title="Gabor" width="50%">
   <figcaption> <strong>Figure 12:</strong> 
  A crater image turned upside down looks like a hill, showing how
  the same image can be generated by different scenes.
  </figcaption>
</figure>


Theoretical neuroscience research suggests that these rules
we use to interpret images are based on the regularities
of the world around us. Our world has a
marked structure that gives rise to many regularities, that is, patterns
that repeat in images and have predictable relationships with the world. For
example, our visual environment is mostly composed of solid objects. Due
to the laws of physics, objects generate continuous contours in images,
consisting of collinear edges. This could explain the rule of
grouping collinear elements as a reasonable
consequence of the world's structure: objects in the world tend
to generate collinear elements, and it would be correct to group them
to form a more global structure.

Similarly, it is easy to notice that in their vast majority,
light sources in our natural environment come from
above, so it makes sense to use a rule
that chooses scene interpretations with this characteristic. Thus, the rules used
by our visual system would be associated with the structure of the world
around us and would help choose the image interpretations that best match it.

But, if these rules are derived from the world's structure to represent
it correctly, why do we perceive the Ames room and other illusions
incorrectly? The answer is that they
are atypical stimuli that violate the normal structure of the world. The
image of the Ames room is generated by a specific perspective on a
room with a very specific irregular shape, but this is a configuration
that is unlikely to occur in the real world. It is much
more likely that an image of that type is generated by a rectangular
room with its parallel walls, ceiling, and floor.
Although these illusions
show the brain's interpretations at work, they occur because of
the use of contrived stimuli that are unlikely to be found in the
real world.

In conclusion, images are fundamentally ambiguous, and
a given image is compatible with infinite possible scenes, so perceiving involves
a process of interpretation. But this process is not arbitrary, as it responds
to rules that fit well with the world's structure, ensuring
that our interpretations are generally good.


So, What Is the Answer to Our Question?
------

We have seen three neurobiological reasons why the world might not be as
we see it: 1) our visual system did not evolve to
allow us to see the world as it is, but to guide
useful actions; 2) representing the world as it is is very
expensive, and 3) the stimuli we receive are ambiguous and require
significant interpretation on our part. In all three cases, we saw
examples that might suggest that the world is not as we see it.
But we also discussed reasons why these limitations are not
determinative: 1) we have very complex behavior that may require a faithful perception
of the world (and available computational research seems to say that our
basic processing is perception is good); 2) we allocate our processing
resources very well, and although a significant part of the world escapes us (which
is inevitable), we build a decent image of it, and 3) our interpretations
use rules that are based on the structure of the
world around us, giving them a solid foundation.

Ultimately, the answer to whether the world is as we see it will depend on
the precise definition of that question and the standards we have for what
constitutes "seeing the world as it is." And if it is
disappointing not to have a concrete answer at the end of the chapter,
perhaps it helps to remember that, despite its simple appearance,
this question has sparked (and continues to spark) discussion throughout a
significant part of the history of ideas.

To conclude, it is worth highlighting that although we marked a distinction
between the philosophical question and the neuroscientific or cognitive one at the beginning
of the chapter, philosophy is an integral part of cognitive sciences and
neuroscience. Many of the questions studied by the latter are based on
concepts that, when deeply examined, lead to analyses and discussions that
currently belong to philosophy. Even though we did not explicitly highlight it,
many points we took for granted in this chapter align with one
or another philosophical perspective and can be harshly criticized from
other philosophical perspectives (e.g., it is debatable whether the visual stimuli we
receive are truly ambiguous, or whether our visual system constructs a
3D model of the world around us). Perhaps a good way to close the chapter
is by revisiting Daniel Dennett (an important philosopher of science), who
eloquently expresses this idea: "There is no such thing as science
free of philosophy, there is only science whose philosophical baggage is carried
on board without examination."

