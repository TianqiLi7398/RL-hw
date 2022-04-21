# Commands in HPRC

The instructions are based on [the guide](https://hprc.tamu.edu/wiki/Grace:QuickStart#Submitting_and_Monitoring_Jobs)

## GOTO workspace

`$ cd $SCRATCH`

### submite jobfile

`$ sbatch xx.slurm`

### check jobs status

`$ squeue -u NetID`

# Run GPU related work [details](https://hprc.tamu.edu/wiki/Grace:QuickStart#Submitting_and_Monitoring_Jobs)

I've made a module called `dl-torch`, just need to load it.

```
  # load all the required modules
  ml purge
          
  # CUDA modules are needed for TensorFlow
  ml GCCcore/9.3.0 GCC/9.3.0 Python/3.8.2 CUDAcore/11.0.2 CUDA/11.0.2 cuDNN/8.0.5.39-CUDA-11.0.2
          
  # As Pytorch comes with CUDA libraries, we don't need to load CUDA modules.
  # the following two modules are sufficient for PyTorch
  # ml GCCcore/10.2.0 Python/3.8.6
          
  # you can save your module list with (dl is an arbitrary name)
  module save dl
       
  # next time when you login you can simply run
  module restore dl
     
  # create a virtual environment (the name dlvenv is arbitrary)
  cd $SCRATCH
  python -m venv dlvenv
  source dlvenv/bin/activate
  ```
