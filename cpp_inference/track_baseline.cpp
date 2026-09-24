// track_baseline.cpp
// Persistence / stationary baselines for cyclone track prediction.
// Metric: great-circle (haversine) error in km, per storm and pooled.
//
// Build: g++ -O2 -std=c++17 track_baseline.cpp -o track_baseline
// Usage: ./track_baseline test.csv [horizon_steps=4] [storm_col=storm_id] [lat_col=lat] [lon_col=lon]
//
// Assumptions (verify against your CSV):
//  - rows for one storm are contiguous and time-ordered
//  - fixed time step between rows (e.g. 6 h) -> horizon_steps * step = lead time
//  - header row present; column names passed via argv if yours differ

#include <algorithm>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <sstream>
#include <string>
#include <vector>

static const double R_EARTH_KM = 6371.0088;
static const double PI = 3.14159265358979323846;

static double deg2rad(double d) { return d * PI / 180.0; }

double haversine_km(double lat1, double lon1, double lat2, double lon2) {
    double p1 = deg2rad(lat1), p2 = deg2rad(lat2);
    double dp = p2 - p1, dl = deg2rad(lon2 - lon1);
    double a = std::sin(dp / 2) * std::sin(dp / 2) +
               std::cos(p1) * std::cos(p2) * std::sin(dl / 2) * std::sin(dl / 2);
    return 2.0 * R_EARTH_KM * std::asin(std::min(1.0, std::sqrt(a)));
}

// wrap longitude difference into [-180, 180]
double wrap_dlon(double d) {
    while (d > 180.0) d -= 360.0;
    while (d < -180.0) d += 360.0;
    return d;
}

struct Point { double lat, lon; };

static std::vector<std::string> split(const std::string& s) {
    std::vector<std::string> out;
    std::string cur;
    std::stringstream ss(s);
    while (std::getline(ss, cur, ',')) {
        // trim whitespace / CR
        cur.erase(0, cur.find_first_not_of(" \t\r\n"));
        cur.erase(cur.find_last_not_of(" \t\r\n") + 1);
        out.push_back(cur);
    }
    return out;
}

struct Stats {
    double sum_pers = 0, sum_stat = 0;
    std::vector<double> errs_pers;
    size_t n = 0;
};

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cerr << "usage: " << argv[0]
                  << " file.csv [horizon_steps=4] [storm_col] [lat_col] [lon_col]\n";
        return 1;
    }
    std::string path = argv[1];
    int H = argc > 2 ? std::stoi(argv[2]) : 4;
    std::string c_storm = argc > 3 ? argv[3] : "storm_id";
    std::string c_lat = argc > 4 ? argv[4] : "lat";
    std::string c_lon = argc > 5 ? argv[5] : "lon";

    std::ifstream f(path);
    if (!f) { std::cerr << "cannot open " << path << "\n"; return 1; }

    std::string line;
    std::getline(f, line);
    auto header = split(line);
    auto idx = [&](const std::string& name) {
        auto it = std::find(header.begin(), header.end(), name);
        if (it == header.end()) {
            std::cerr << "column '" << name << "' not found. header:\n  " << line << "\n";
            std::exit(1);
        }
        return (int)(it - header.begin());
    };
    int i_s = idx(c_storm), i_la = idx(c_lat), i_lo = idx(c_lon);

    // storm -> ordered track (order of appearance preserved)
    std::vector<std::string> order;
    std::map<std::string, std::vector<Point>> tracks;
    while (std::getline(f, line)) {
        if (line.empty()) continue;
        auto c = split(line);
        if ((int)c.size() <= std::max({i_s, i_la, i_lo})) continue;
        try {
            Point p{std::stod(c[i_la]), std::stod(c[i_lo])};
            if (!tracks.count(c[i_s])) order.push_back(c[i_s]);
            tracks[c[i_s]].push_back(p);
        } catch (...) { continue; }  // skip malformed rows
    }

    std::cout << std::fixed << std::setprecision(1);
    std::cout << "Horizon = " << H << " steps\n";
    std::cout << std::left << std::setw(20) << "storm" << std::setw(8) << "n"
              << std::setw(16) << "persistence_km" << "stationary_km\n";

    Stats all;
    for (const auto& sid : order) {
        const auto& t = tracks[sid];
        Stats s;
        // need t[i-1], t[i], t[i+H]
        for (size_t i = 1; i + H < t.size(); ++i) {
            double dlat = t[i].lat - t[i - 1].lat;
            double dlon = wrap_dlon(t[i].lon - t[i - 1].lon);
            // persistence: last per-step displacement continues H more steps
            double plat = t[i].lat + H * dlat;
            double plon = t[i].lon + H * dlon;
            const Point& truth = t[i + H];
            double e_p = haversine_km(plat, plon, truth.lat, truth.lon);
            double e_s = haversine_km(t[i].lat, t[i].lon, truth.lat, truth.lon);
            s.sum_pers += e_p; s.sum_stat += e_s; s.errs_pers.push_back(e_p); s.n++;
            all.sum_pers += e_p; all.sum_stat += e_s; all.errs_pers.push_back(e_p); all.n++;
        }
        if (s.n == 0) {
            std::cout << std::setw(20) << sid << "too short for this horizon\n";
            continue;
        }
        std::cout << std::setw(20) << sid << std::setw(8) << s.n
                  << std::setw(16) << s.sum_pers / s.n << s.sum_stat / s.n << "\n";
    }

    if (all.n == 0) { std::cout << "no valid samples\n"; return 0; }
    auto v = all.errs_pers;
    std::sort(v.begin(), v.end());
    std::cout << "\nPOOLED  n=" << all.n
              << "  persistence mean=" << all.sum_pers / all.n
              << " km  median=" << v[v.size() / 2]
              << " km  |  stationary mean=" << all.sum_stat / all.n << " km\n";
    std::cout << "Note: pooled mean weights long storms more. Report per-storm numbers too.\n";
    return 0;
}