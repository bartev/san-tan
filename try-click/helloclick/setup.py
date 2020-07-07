from setuptools import setup


setup(
    name='HelloWorld',
    version='1.0',
    py_modules=['hello'],
    install_requires=[
        'Click', ],
    entry_points='''
      [console_scripts]
      hello=hello:foo
      findme=multval:findme
      putitem=multval:putitem
      putitem2=multval:putitem2
      commit=multval:commit
      logme=multval:logme
      info2=multval:info2
      info3=multval:info3
      trans=multval:trans
      digest=multval:digest
      hello_name=multval:hello_name
      encrypt=multval:encrypt
      encrypt2=multval:encrypt2
      hello_un=multval:hello_un
      get_version=multval:get_version
      dropdb1=multval:dropdb1
      greet=env_var:greet
      greet2=env_var:greet2
      greet3=env_var:greet3
      perform=env_var:perform
      usecontext=use_context:cli
   ''',
)
