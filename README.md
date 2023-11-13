# GANs for HEP Project


### Git practices

To clone in perlmutter:
```
~> git clone $CFS/m3443/data/ForHadronic . -r
```
To stage (you can pick and choose which to stage to be committed in the next step):
```
ForHadronic> git add <files>
```
To commit:
```
ForHadronic> git commit -m "..."
```
If I'm not mistaken, your commits will be reflected in the communal folder.

To clone on local machine:
```
> git clone perlmutter.nersc.gov:/global/cfs/cdirs/m3443/data/ForHadronic -r .
```
or
```
> git clone <hostname>@perlmutter.nersc.gov:/global/cfs/cdirs/m3443/data/ForHadronic -r .
```
I think this should work. Have sortof tested it on my own computer.

#### Some notes
- Please pull often, we might get nasty conflicts that you'll have to work through if you don't.
- Please make smaller commit chunks, ideally isolated chunks of work/tasks so that reading back on commits is easy
- Use a git grapher if you want to visualize the commits. VSCode has a 'Git Graph' extension or use Sublime Merge
- If you have a new proposal that may break things, create a new branch `git checkout <branch name>`. This creates an copy where you can create your own implementation without affecting the master branch.
- Please keep the master branch clean and working. It should be what we fall back on when things don't work.


I'm not sure how much we should follow CI/CD pipelines (creating testing, beta, alphas, releases?)


## Linux commands
```
top
free
uname -a
squeue -u [userid]
showquota
```