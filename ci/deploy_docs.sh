#!/bin/bash
set -e # exit with nonzero exit code if anything fails

# Clone *this* git repo, but only the gh-pages branch. We redirect any output to
# /dev/null to hide any sensitive credential data that might otherwise be exposed.
if [[ ! -d $GH_PAGES_DIR ]]; then
    git clone -q -b gh-pages --single-branch https://${GH_TOKEN}@github.com/${TRAVIS_REPO_SLUG}.git $GH_PAGES_DIR 2>&1 >/dev/null
fi
cd $GH_PAGES_DIR

# inside this git repo we'll pretend to be a new user
git config user.name "Travis CI"
git config user.email "travis@nobody.org"

if [[ "${TRAVIS_TAG}" != "" ]]; then
    export VERSION=${TRAVIS_TAG%.*}
else
    export VERSION=dev
fi

# The first and only commit to this new Git repo contains all the
# files present with the commit message "Deploy to GitHub Pages".
rm -rf ${VERSION}
cp -R ${TRAVIS_BUILD_DIR}/docs/build/html/ ${VERSION}/
touch .nojekyll
if [[ "${VERSION}" != "dev" ]]; then
    ln -shf ${VERSION} latest
fi

git add -A .
if [[ "${VERSION}" == "dev" && `git log -1 --format='%s'` == *"dev"* ]]; then
    git commit --amend --reset-author --no-edit
else
    git commit -m "Deploy ${VERSION} to GitHub Pages"
fi

# Push up to gh-pages
echo Pushing...
git push --force --quiet origin gh-pages
