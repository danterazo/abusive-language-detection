# Documentation
I tried to make this as automated and modular as possible so I could run it on the NLP server
with `nohup`. This requires hardcoded inputs. I iterate over lists/arrays of inputs so that the
multiple models can be trained consecutively. This means that I effectively "set it and
forget it." With the size of the Kaggle dataset it's important to check `nohup` output for progress often!

**REMEMBER: It's okay to fail in research!**