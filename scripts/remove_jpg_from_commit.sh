for i in `git status --porcelain | grep '\.jpg$' | sed 's/...//'`; do
    echo "$i"    
    git reset HEAD "$i"
##    git checkout "$i"
done
