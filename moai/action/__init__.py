try:
    from moai.action.train import train
    from moai.action.play import play
    from moai.action.evaluate import evaluate
    from moai.action.plot import plot
    from moai.action.diff import diff
    from moai.action.reprod import reprod
except:
    from action.train import train
    from action.play import play
    from action.evaluate import evaluate
    from action.plot import plot
    from action.diff import diff
    from action.reprod import reprod