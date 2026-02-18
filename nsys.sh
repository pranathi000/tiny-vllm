rm -rf report1.nsys-rep
rm -rf repor1.sqlite
./test.sh
nsys profile ./build/tiny-vllm
nsys stats report1.nsys-rep