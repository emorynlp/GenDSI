import ezpyz as ez
import fabric as fab
import pathlib as pl


def download(machine, path):
    hosts = dict(
        tebuna='localhost:55555',
        h100='localhost:55556'
    )
    password = ez.File(pl.Path.home()/'.pw/emory').read().strip()
    project = pl.Path('/local/scratch/jdfinch/dstrr/')
    credentials = dict(
        user='jdfinch',
        connect_kwargs=dict(password=password)
    )
    with fab.Connection(hosts[machine], **credentials) as conn:
        path = pl.Path(path)
        is_folder = conn.run(
            f'test -d {project/path} && echo 1 || echo 0'
        ).stdout.strip() == '1'
        if is_folder:
            tar_file = path.with_suffix('.tar.gz')
            conn.run(f'cd {project} && tar -czvf {tar_file} {path}')
            conn.get(f'{project / tar_file}', str(tar_file))
            conn.run(f'rm {project / tar_file}')
            conn.local(f'tar -xzvf {tar_file}')
            conn.local(f'rm {tar_file}')
        else:
            conn.get(f'{project/path}', path)





if __name__ == '__main__':
    download('h100', 'data/sgd_wo_domains/valid_DSG/')