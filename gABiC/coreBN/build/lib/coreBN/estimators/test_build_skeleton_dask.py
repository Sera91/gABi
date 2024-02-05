import networkx as nx
from kernel_CITests_new import kernel_CItest

def build_skeleton_from_data(input_file, ci_test, variant, ):

	if ci_test == "hsic_gamma":
            sel_method="hsic.gamma"
        elif ci_test == "hsic_perm":
            sel_method="hsic.perm"
        elif ci_test == "dcc_perm":
            sel_method="dcc.perm"
        elif ci_test == "dcc_gamma":
            sel_method="dcc.gamma"
        else:
            raise ValueError(
                f"ci_test must be either hsic_gamma, hsic_perm or dcc_perm or a function. Got: {ci_test}")

	graph = nx.complete_graph(n=variables, create_using=nx.Graph)

	if variant == "dask-smart":

            lim_neighbors = 0
            separating_sets = dict()
            # Exit condition: 1. If all the nodes in graph has less than `lim_neighbors` neighbors.
            #             or  2. `lim_neighbors` is greater than `max_conditional_variables`.
            while not all([len(list(graph.neighbors(var))) <lim_neighbors for var in self.variables]):

                def _dask_parallel_fun(u, v, input_data, sel_method):
                    for separating_set in chain(
                        combinations(set(graph.neighbors(u)) -
                                     set([v]), lim_neighbors),
                        combinations(set(graph.neighbors(v)) -
                                     set([u]), lim_neighbors),
                    ):
                        if kernel_CItest(
                            u,
                            v,
                            separating_set,
                            data=input_data,
                            method=sel_method,
                            boolean=True,
                            significance_level=significance_level,
                            **kwargs,
                        ):
                            return (u, v), separating_set # the return breaks the for cycle 
	                    # Note that in the case where kernel_CI_test is false 
	                    # this function will return None

                # We expect SLURM_NTASKS-2 workers
                #N = int(os.getenv('SLURM_NTASKS'))-2

                # Wait for these workers and report
                client.wait_for_workers()# n_workers=N

                num_workers = len(client.scheduler_info()['workers'])
                print("%d workers available and ready" % num_workers)

                if n_jobs > N:
                    n_jobs = N

                
           
		futures =                 
		futures2 = client.map(_dask_parallel_fun, graph.edges())   
                for future, result in as_completed(futures, with_results=True):
                    if result is not None:
                        (u, v), sep_set = result
                        graph.remove_edge(u, v)
                        separating_sets[frozenset((u, v))] = sep_set

                # Step 3: After iterating over all the edges, expand the search space by increasing the size
                #         of conditioning set by 1.
                if lim_neighbors >= max_cond_vars:
                    logging.info(
                            "Reached maximum number of allowed conditional variables. Exiting"
                    )
                    break
                lim_neighbors += 1

                if show_progress and SHOW_PROGRESS:
                    pbar.update(1)
                    pbar.set_description(
                        f"Working for n conditional variables: {lim_neighbors}"
                    )

        else:

