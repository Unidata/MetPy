# Set repo info
REPO_URL="https://github.com/Unidata/MetPy.git" #metpy repo to clone from
CLONE_DIR_GEN_HASH="temp_repo_generate_hashes" #temporary repo to clone to - deleted at the end of the script

# clone metpy and fetch tags
git clone --depth=100 --no-tags "$REPO_URL" "$CLONE_DIR_GEN_HASH" #shallow clone metpy repo
cd "$CLONE_DIR_GEN_HASH" || exit 1 #change directories to temporary repo, if this fails exit
git fetch --tags #fetch metpy tags

# Set the range: from last v1.6.x to present (all 1.7.x merge commits) - no commits authored by or mentioning dependabot or github-actions
git log --merges v1.6.3.. --pretty=format:"%H %s" | \
grep -v -i "dependabot" | \
grep -v -i "github-actions" | \
awk '{print $1}'  > ../no_bot_merge_commits.txt #print output to this file in the benchmarks dir


#Get the commit hashes for each minor version after 1.
git for-each-ref --sort=version:refname \
  --format='%(refname:short) %(objectname)' refs/tags | \
  grep -E '^v[1-9].[4-9]*\..*' |
  awk '{print $2}' >> ../no_bot_merge_commits.txt #append these results to same file 
  
cd .. #leave temp_repo

rm -rf "$CLONE_DIR_GEN_HASH" #clean up by removing temporary repo