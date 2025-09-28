# xTrack Sensys - Person Tracking Project

This project focuses on the identification, localization, and tracking of people using sensor data from the xTrack vehicle. The primary goal is to detect and track individuals, with a special focus on those wearing yellow safety vests, using data from an RGB-D camera and LiDAR.

## Documentation

This project includes comprehensive Sphinx documentation. To build and view the documentation:

### Building Documentation

```bash
# Build the documentation
sphinx-build -b html docs/source docs/build
```

The built documentation will be available in `docs/build/html/index.html`.

### Documentation Contents

The documentation includes:

- **Getting Started Guide**: Installation, setup, and basic usage
- **User Guide**: Detailed command-line options and configuration
- **API Reference**: Complete documentation of all modules and functions
- **Testing Guide**: Comprehensive testing and benchmarking procedures
- **Troubleshooting**: Common issues and solutions

## Code Quality

This project uses `ruff` for code formatting and linting:

```bash
ruff format . && ruff check . --fix
```

## License

This project is licensed under the GPLv3 License - see the [LICENSE](LICENSE) file for details.
