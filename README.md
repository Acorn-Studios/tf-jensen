
# Jensen (TF2 Demo AI) 🔮
Jensen is a project to complete the vision of [megascatterbomb](https://www.youtube.com/@megascatterbomb)'s ML cheater detector under the same name. 

We use 2 methods for digesting demos and collecting cheaters.
> Use [megascatterbomb](https://www.youtube.com/@megascatterbomb)'s rust [anticheat](https://github.com/MegaAntiCheat) template to detect the spots where cheaters cheat, and then validate that they are actually cheating to make a smaller and more refined model
> 
> Digest entire demos and spit out cheaters. This is more ambitious but should be able to detect blatant esp and other cheats with enough data. The data is also easier to collect and requires less human input

These models have different classifications: `jensen-vigil` and `jensen-nightwatch` respectively.
See the below sections for how each one works, how to use it, etc.

## Methods

### With `jensen-vigil`
Here's the broad idea:

**Usage**
> Use normal [megaanticheat](https://github.com/MegaAntiCheat) cheat detection to find ticks with cheating activity
> 
> Narrow the ticks to only our target after getting the `viewangles` as a csv
> 
> Feed those into a `text to 1/0 model` (repurposed from sentiment analysis project) with a 10 tick buffer on each side to improve accuracy (21 deg total, should be enough for fast flicks)
> 
> Convict the cheater of cheating (`1`) or being legit (`0`)

**Training**
> Gather demos with cheaters
> 
> Get ticks where they were cheating
> 
> Get a human to report if there were suspicious flicks at those angles & feed that into our training data
> 
> Clean data & train our model!

Overall data gathering ease: *medium*, medium

### With `jensen-nightwatch`
Here's the gist:

**Usage**
> Get the csv of a demo file like before
> 
> Feed that into `jensen-nightwatch` to get a list of cheater's steam ids
> 
> Convict cheater!

**Training**
> Gather demos with cheaters 
> 
> Label all the cheaters in the demo
> 
> Feed all of the position and viewangle data into our model and train it with the outputs of their steam ids

Overall data gathering ease: *high*, easy


