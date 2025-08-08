---
title: 'GitHub releases to keep your old projects from breaking'
date: 2024-09-25
permalink: /posts/2024/09/github-releases-to-keep-old-projects-from-breaking/
tags:
  - GitHub
  - Programming practices
  - Project management
---

If you are a scientist programming for your research, you probably
experienced the following. You wrote some code,
in a directory `my_code/`, or GitHub repository
`https://github.com/my_user/my_code`. You use this code in a
project, and analyses are working. Then, you
modify your software to add a new feature, required by a new
project or analysis. However, the new modified code
might (and probably will) break the original analysis.

Of course, you don't want your old project to now become
unreplicable because of these changes. But you also
don't want to make the old project work with the new code,
which can involve a lot of work.
One way to avoid this situation is to
use [GitHub releases](https://docs.github.com/en/repositories/releasing-projects-on-github/about-releases).

What are GitHub releases?
---------------------

Here we'll assume that you are familiar with
the very basics of GitHub, and that you know
how to create a repository, commit and push code.
If not, consider reading this
[guide for scientists in PLoS Comp Bio](https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1004668)
to start incorporating this amazing tool into your workflow.

The basic GitHub workflow of committing and pushing
does not protect you from the situation described
above, however. If you push changes to the repository,
the old project will use the new code and might break.

GitHub releases basically allow you to "freeze" a version
of your code that you want to be able to access
later (e.g. the version that works with the old project),
while still letting you work on the code and make
changes. You have probably seen this in many
software projects, where you can download a
specific version of the software, like
`useful_package_v1.0.0`, `useful_package_v1.0.1`, etc.
GitHub releases allow you to do the same thing
with your code, so that for different projects you
can specify which version of the code you and your
users should use.

How to use GitHub releases
-----------------------

Using GitHub releases is quite simple, and the
[GitHub documentation](https://docs.github.com/en/repositories/releasing-projects-on-github/managing-releases-in-a-repository)
is clear and helpful.

Once you have the desired working version of your code
in your repository, all you need to do is go
to your repository, click on the "Releases" tab
on the right. Then a page to create the new release
will open up. You can give the release a tag (e.g. `v1.0.0`),
a title (e.g. `First release`), and a description (e.g.
`Project X works with this release`).

After you create the release, you can go back to the
"Releases" tab and see all the releases you have created.
You can download the code for a specific release
using the command line
```
git clone https://github.com/my_user/my_packave --branch v1.0.0
```
In the repository of your older project, you can now
direct users to download the code from the release
that works with the project by using this command.
If you already have a `environment.yml` file that
you use to set up the environment for the project,
you can specify the version of the code that should be
used by adding the version tag to the URL, like this:
```
- git+https://github.com/my_user/my_package.git@v1.0.0
```

You can also just download the code by clicking on the release
and then on the "Source code" link. 

