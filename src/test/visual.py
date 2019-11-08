from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib.cm as cm


def plot_building(ax, buildings, weight=0.5, style='-'):
    for i in tqdm(buildings):
        ax.add_patch(plt.Polygon(i.polygon.exterior.coords,
                                 fill=False, ec="#b0b0b0", lw=weight, ls=style))


def plot_frame(ax, frames):
    for i in tqdm(frames):
        line_fliped = list(map(tuple, zip(*i.line.coords)))
        ax.plot(line_fliped[0], line_fliped[1], 'm-', lw=1)
        ax.text(i.line.centroid.x, i.line.centroid.y, f'{i.id}')


def plot_section_point(ax, frames):
    for frame in tqdm(frames):
        if len(frame.observers) == 0:
            pass
        else:
            tmp = [(j.position.x, j.position.y) for j in frame.observers]
            fliped = list(map(tuple, zip(*tmp)))
            ax.scatter(fliped[0], fliped[1], c='r', s=3)
            for idx, o in enumerate(frame.observers):
                ax.text(o.position.x, o.position.y, "{}:{}".format(frame.id, o.id), c='r')


def plot_section_line(ax, frames):
    for i in tqdm(frames):
        if len(i.secline) == 0:
            pass
        else:
            for j in i.secline:
                fliped = list(map(tuple, zip(*j.coords)))
                ax.plot(fliped[0], fliped[1], 'r-', lw=1)
                # ax.scatter(*j.coords[0], marker='o', s=5, c='w')
                # ax.scatter(*j.coords[-1], marker='o', s=5, c='k')


def plot_isovist_sec_by_pt(ax, frames):
    area_isovist = []
    plotpt_x = []
    plotpt_y = []
    texts = []
    for line in tqdm(frames):
        # framelineに分割点がなかった場合
        if isinstance(line.isovist_sec, type(None)):
            pass
        else:
            isovists = [_ for _ in line.isovist_sec if not isinstance(_.positon, type(None))]
            if len(isovists) == 0:  # 分割面で道路が見つからなかった場合
                pass
            else:
                for i in isovists:
                    area_isovist.append(i.isovist.area)
                    plotpt_x.append(i.positon.x)
                    plotpt_y.append(i.positon.y)
                    texts.append("{}:{}".format(line.id, i.id))
    for i, area in enumerate(area_isovist):
        ax.scatter(plotpt_x[i], plotpt_y[i], color=cm.Spectral(area / max(area_isovist)))
        ax.text(plotpt_x[i], plotpt_y[i], texts[i], c='b')


def plot_isovist_sec_polygon(ax, frames, idx_frame, idx_isovist, alpha=1):
    observer = frames[idx_frame].observers[idx_isovist]
    poly = observer.isov_sec.polygon
    ax.add_patch(plt.Polygon(poly.boundary.coords, alpha=alpha))
    ax.set_xlim(-observer.sight, observer.sight)
    ax.set_ylim(0, observer.sight)


def plot_building_in_circle(circle, buildings, scan):
    crv_building = [i.polygon.boundary.coords for i in buildings]
    
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    for i in crv_building:
        poly = plt.Polygon(i, fc="#868686")
        tmp = list(map(list, zip(*i)))
        ax.add_patch(poly)
    cir = plt.Polygon(circle.coords, fc='none', ec="#000000", ls='--')
    ax.add_patch(cir)
    for i in scan.vertex:
        tmpx = [scan.positon.x, i[0]]
        tmpy = [scan.positon.y, i[1]]
        ax.plot(tmpx, tmpy, c='k', ls='--')
    
    bndry = list(map(list, zip(*circle.coords)))
    blank = (max(bndry[0]) - min(bndry[0])) * 0.3
    ax.set_xlim(min(bndry[0]) - blank, max(bndry[0]) + blank)
    ax.set_ylim(min(bndry[1]) - blank, max(bndry[1]) + blank)
    ax.set_aspect('equal')
    
    plt.show()
