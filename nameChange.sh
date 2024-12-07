for f in *; do
    if [ -d "$f" -a "$f" != "data" ]; then
        # Will not run if no directories are available
        echo $f
        for z in "$f/2*.png"; do
            mv $z "$f/MNIST.png"
        done
        for z in "$f/1*.png"; do
            mv $z "$f/CIFAR10.png"
        done
    fi
done