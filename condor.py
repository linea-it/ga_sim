from parsl.config import Config
from parsl.monitoring.monitoring import MonitoringHub
from parsl.providers import CondorProvider
from parsl.providers import LocalProvider
from parsl.executors import HighThroughputExecutor
from parsl.addresses import address_by_hostname
import os


def get_config(key='local'):
    """
    Creates an instance of the Parsl configuration 

    Args:
        phz_config (dict): Photo-z pipeline configuration - available in the config.yml
    """
    ga_sim_root_dir = os.getenv('GA_ROOT')

    executors = {
        "htcondor": HighThroughputExecutor(
            label='htcondor',
            address=address_by_hostname(),
            max_workers=30,
            provider=CondorProvider(
            init_blocks=3,
            min_blocks=3,
            max_blocks=3,
            # parallelism=0.5,
            requirements='''((Machine == "apl05.ib0.cm.linea.gov.br")||(Machine == "apl06.ib0.cm.linea.gov.br")||(Machine == "apl07.ib0.cm.linea.gov.br")||(Machine == "apl08.ib0.cm.linea.gov.br")||(Machine == "apl09.ib0.cm.linea.gov.br")||(Machine == "apl10.ib0.cm.linea.gov.br")||(Machine == "apl11.ib0.cm.linea.gov.br")||(Machine == "apl12.ib0.cm.linea.gov.br")||(Machine == "apl13.ib0.cm.linea.gov.br")||(Machine == "apl14.ib0.cm.linea.gov.br")||(Machine == "apl15.ib0.cm.linea.gov.br")||(Machine == "apl15.ib0.cm.linea.gov.br"))''',
            scheduler_options='+RequiresWholeMachine = True',
            worker_init=f"source {ga_sim_root_dir}/env.sh",
            cmd_timeout=120,
            ),
        ),
        "local": HighThroughputExecutor(
            label='local',
            max_workers=20,
            provider=LocalProvider(
                min_blocks=1,
                init_blocks=1,
                max_blocks=1,
                nodes_per_block=1,
                parallelism=0.5
            )
        )
    }

    return Config(
        executors=[executors.get(key)],
        strategy=None
    )
