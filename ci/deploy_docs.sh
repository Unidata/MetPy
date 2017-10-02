#!/bin/bash
set -e # exit with nonzero exit code if anything fails

# Decrypt and activate the deploy key
echo Setting up access...
openssl aes-256-cbc -K $encrypted_091b7ae1977a_key -iv $encrypted_091b7ae1977a_iv -in ${TRAVIS_BUILD_DIR}/ci/deploy_key.enc -out ${TRAVIS_BUILD_DIR}/ci/deploy_key -d
chmod 600 ${TRAVIS_BUILD_DIR}/ci/deploy_key
eval `ssh-agent -s`
ssh-add ${TRAVIS_BUILD_DIR}/ci/deploy_key

# Clone *this* git repo, but only the gh-pages branch.
echo Cloning gh-pages...
if [[ ! -d $GH_PAGES_DIR ]]; then
    git clone -q -b gh-pages --single-branch git@github.com:${TRAVIS_REPO_SLUG}.git $GH_PAGES_DIR
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
echo Updating $VERSION docs...
rm -rf ${VERSION}
cp -R ${TRAVIS_BUILD_DIR}/docs/build/html/ ${VERSION}/
touch .nojekyll
if [[ "${VERSION}" != "dev" ]]; then
    ln -snf ${VERSION} latest
fi

# Generate our json list of versions
echo Generating versions.json...
${TRAVIS_BUILD_DIR}/ci/gen_versions_json.py

echo Staging...
git add -A .
if [[ "${VERSION}" == "dev" && `git log -1 --format='%s'` == *"dev"* ]]; then
    git commit --amend --reset-author --no-edit
else
    git commit -m "Deploy ${VERSION} to GitHub Pages"
fi

# Push up to gh-pages
echo Pushing...
git push --force origin gh-pages
