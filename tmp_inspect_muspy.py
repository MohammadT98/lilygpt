import muspy
import pkgutil

print('muspy_version=', getattr(muspy, '__version__', None))
print('muspy_file=', getattr(muspy, '__file__', None))

# list top-level attributes containing likely metric names
keys = [k for k in dir(muspy) if any(t in k for t in ('note','pitch','entropy','repet','ioi','density','average','n_p','range','metric','feature','analysis'))]
print('candidate_keys=', keys)

# list submodules
if hasattr(muspy, '__path__'):
    mods = [m.name for m in pkgutil.iter_modules(muspy.__path__)]
    print('submodules=', mods)
else:
    print('no __path__')
