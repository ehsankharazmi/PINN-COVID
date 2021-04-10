# Identifiability and predictability of <br/> integer- and fractional-order epidemiological models <br/> using physics-informed neural networks

We analyze a plurality of epidemiological models through the lens of physics-informed neural networks (PINNs) that enable us to identify multiple time-dependent parameters and to discover new data-driven fractional differential operators. In particular, we consider several variations of the classical susceptible-infectious-removed (SIR) model by introducing more compartments and delay in the dynamics described by integer-order, fractional-order, and time-delay models.

## PINN-COVID
The `PINN-COVID` is a Python package containing tools for studying identifiability, predictibility, and uncertainty quantification of epidemiological models. The codes require only a standard computer with enough RAM and CPU/GPU computation power.

### Python Dependencies
We use the Python 3.7.4 and `PINN-COVID` mainly depends on the following Python packages.

```
matplotlib==3.3.3
numpy==1.19.5
pandas==1.2.1
pyDOE==0.3.8
scipy==1.6.0
tensorflow-gpu==1.15.0
```

## Running the Codes

In each folder for different datasets, there are two Python codes for each model: one with the term `_training` and one with the term `_PostProcess`. The `training` code runs the corresponding model to infere the parameters/dynamics and saves the results. Each time the code is run, the results are saved in a separate folder. The `PostPocess` code computes the mean and std of the results from `training`.  The  computationally expensive part of the simulation is running the  `training` codes, especially that for each case and each dateset, we run this code at least 10 times to compute the corresponding uncertainties. The run time strongly depends on the computational power of the machine on which the code is running.

### Running the Codes on a New Dataset

The PINN-COVID has the flexibility to be run on a new dataset. If the available data on the epidemiological classes are the same as the ones considered in the codes, then only the loading part of the code should be modified. However, If the available data includes/excludes certain epidemiological classes, then specific modifications should be made in the loss function of the network to accommodate  the new dataset. We refer to the Supplementary Information provided in the Reference. 


## Reference

The detail explanation of the formulation and results can be found here: [![Reference](https://www.medrxiv.org/content/10.1101/2021.04.05.21254919v1)](https://www.medrxiv.org/content/10.1101/2021.04.05.21254919v1)


## License

This project is covered under the **MIT License**.
