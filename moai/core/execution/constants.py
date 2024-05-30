
class Constants(object):
    # MOAI EXECUTION
    _MOAI_: str = '_moai_'
    # _EXECUTION_: str = f'{_MOAI_}._execution_'
    # _COLLECTIONS_: str = f'{_MOAI_}._collections_'
    # _FLOWS_: str = f'{_MOAI_}._flows_'
    # _MONITORING_: str = f'{_MOAI_}._monitoring_'
    _MOAI_MONITORING_: str = f'{_MOAI_}._monitoring_'
    _MOAI_METRICS_: str = f'{_MOAI_}._metrics_'
    _MOAI_LOSSES_: str = f'{_MOAI_}._losses_'
    _MOAI_LOSSES_RAW_: str = f'{_MOAI_}._losses_.raw'
    _MOAI_LOSSES_WEIGHTED_: str = f'{_MOAI_}._losses_.weighted'
    _MOAI_LOSSES_TOTAL_: str = f'{_MOAI_}._losses_.total'
    _MOAI_EXECUTION_: str = f'{_MOAI_}._execution_'
    _MOAI_INITIALIZE_: str = f'{_MOAI_}._initialize_'
    _EXECUTION_: str = f'_execution_'
    _COLLECTIONS_: str = f'_collections_'
    _METRICS_COLLECTION_: str = f'{_COLLECTIONS_}._metrics_'
    _OBJECTIVES_COLLECTION_: str = f'{_COLLECTIONS_}._objectives_'
    _MONITORS_COLLECTION_: str = f'{_COLLECTIONS_}._monitors_'
    _CRITERIA_COLLECTION_: str = f'{_COLLECTIONS_}._criteria_'
    _OPTIMIZERS_COLLECTION_: str = f'{_COLLECTIONS_}._optimizers_'
    _SCHEDULERS_COLLECTION_: str = f'{_COLLECTIONS_}._schedulers_'
    _FLOWS_: str = f'_flows_'
    _MONITORING_: str = f'_monitoring_'
    _SCHEDULE_: str = f'_schedule_'

    # MODEL
    _MODEL_: str = 'model'

    _PARAMETERS_: str = 'parameters'
    _CRITERIA_: str = f'criteria'

    _PARAMETER_GROUPS_: str = f'{_PARAMETERS_}.groups'
    _PARAMETER_OPTIMIZERS_: str = f'{_PARAMETERS_}.optimizers'
    _PARAMETER_INITIALIZERS_: str = f'{_PARAMETERS_}.initializers'
    _PARAMETERS_CRITERIA_: str = f'{_PARAMETERS_}.criteria'
    _PARAMETER_SCHEDULERS_: str = f'{_PARAMETERS_}.schedulers'
    _OPTIMIZERS_: str = f'optimizers'
    _INITIALIZERS_: str = f'initializers'
    _INITIALIZE_: str = f'_initialize_'
    _CRITERIA_: str = f'criteria'
    _GROUPS_: str = f'groups'
    _SCHEDULERS_: str = f'schedulers'

    # _OBJECTIVE_: str = 'objective'
    # _TERMS_: str = f'{_MODEL_}.{_OBJECTIVE_}.terms'

    # _VALIDATION_: str = 'validation'
    # _METRICS_: str = f'{_MODEL_}.{_VALIDATION_}.metrics'

    _OBJECTIVES_: str = f'objectives'
    
    _METRICS_: str = f'{_MODEL_}.metrics'

    # _TERMINATION_: str = 'termination'
    # _CRITERIA_: str = f'{_MODEL_}.{_TERMINATION_}.criteria'

    _MONITORS_: str = f'{_MODEL_}.monitors'

    _MODULES_: str = f'{_MODEL_}.modules'
    
    _MONADS_: str = f'{_MODEL_}.monads'

    _REMODEL_: str = f'{_MODEL_}.remodel'

    # ENGINE
    _ENGINE_: str = 'engine'

    _RUNNER_: str = f'{_ENGINE_}.runner'
    _MODULES_: str = f'{_ENGINE_}.modules'
    _LOGGERS_: str = f'{_ENGINE_}.loggers' # or directly in runner?

    # DATA
    _DATA_: str = 'data'

    _TRAIN_DATA_: str = f'{_DATA_}.train'
    _VALIDATION_DATA_: str = f'{_DATA_}.val'
    _TEST_DATA_: str = f'{_DATA_}.test'
    _PREDICT_DATA_: str = f'{_DATA_}.predict'
