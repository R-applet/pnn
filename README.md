# pnn
Genetic algorithm to find the simplest interpretable model for a given problem.

The notebook titled EnergMats_KJ_v1.ipynb is the structure of the PNNs with KJ inputs. EnergMats_atomic_v2.ipynb is the structure of the PNNs with C, H, N, O atomic fractions, heats of formation (not heat of detonations) and density as inputs. ExpData_v2.csv contains all 190 explosives and the CHNO_data.csv data set contains (149) only those that contain either C, H, N, O but not any other elements. The py files pnn_kj.py and pnn_atomic.py will run the full genetic algorithm for each structure. Eqn_extract.ipynb is a notebook that can be used to extract an equation from an individual. EvolutionAnalysis.ipynb is a notebook with a few examples of generating the pareto front.
